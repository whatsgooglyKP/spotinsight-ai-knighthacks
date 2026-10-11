"""
RoadGuard Agent: Boston Dynamics SPOT Robot Payload & Edge Inference Daemon.
Simulates onboard SPOT payload computer (Core I/O / Jetson Orin) executing
autonomous autowalk missions, waypoint navigation, and real-time fault triage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import time
import pandas as pd

from roadguard.config import LAKE_COUNTY_WAYPOINTS
from roadguard.models.classifier import RoadGuardPredictor
from roadguard.models.vision_features import VisionFeatureExtractor


@dataclass
class SpotEdgeTelemetry:
    """Real-time SPOT payload telemetry state."""
    mission_id: str
    action_id: str
    current_waypoint: str
    corridor_name: str
    battery_pct: float
    x_coord: float
    y_coord: float
    payload_status: str
    edge_prediction: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class SpotPayloadEdgeService:
    """
    Simulates Boston Dynamics SPOT payload software architecture:
    - Reads mission actions & replay events
    - Executes local edge classifier inference
    - Detects critical battery drain & path obstruction
    - Broadcasts high-priority alerts and AR HUD packets
    """

    def __init__(self, predictor: Optional[RoadGuardPredictor] = None):
        self.predictor = predictor or RoadGuardPredictor()
        self.vision_extractor = VisionFeatureExtractor()
        self.telemetry_history: List[SpotEdgeTelemetry] = []
        self.alert_subscribers: List[Callable[[Dict[str, Any]], None]] = []

    def register_alert_subscriber(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribes downstream services (e.g., AR Glasses HUD, Municipal Dispatch) to edge alerts."""
        self.alert_subscribers.append(callback)

    def process_incoming_action(
        self, action_record: Dict[str, Any], image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes edge inference cycle on incoming SPOT action telemetry.
        """
        loc_id = action_record.get("location_id", "WP01")
        loc_meta = LAKE_COUNTY_WAYPOINTS.get(loc_id, {})

        # Enrich action record with Lake County attributes if missing
        enriched_action = dict(action_record)
        if "lake_county_aadt" not in enriched_action:
            enriched_action["lake_county_aadt"] = loc_meta.get("annual_daily_traffic", 20000)
            enriched_action["speed_limit_mph"] = loc_meta.get("speed_limit_mph", 45)
            enriched_action["lake_county_karst_risk"] = loc_meta.get("karst_subsidence_risk", "Moderate")
            enriched_action["lake_county_flood_zone"] = loc_meta.get("flood_zone", "X")
            enriched_action["lake_county_corridor"] = loc_meta.get("corridor_name", "Lake County Roadway")

        # Extract lightweight computer vision features if inspection image is provided
        cv_features = {}
        if image_path:
            cv_features = self.vision_extractor.extract_features_from_image(image_path)
            # Update visual reading proxy from CV distress score if visual reading is absent
            if "reading_visual_score" not in enriched_action or pd.isna(enriched_action.get("reading_visual_score")):
                enriched_action["reading_visual_score"] = max(10.0, 100.0 - cv_features["cv_overall_distress_score"])

        # Execute local supervised model inference on SPOT edge payload
        prediction = self.predictor.predict_action(enriched_action)

        # Evaluate edge safety triggers
        battery = float(enriched_action.get("battery_pct", 80.0))
        is_major = prediction["major_liability_prediction"] == 1
        requires_immediate_action = is_major or battery < 20.0

        telemetry = SpotEdgeTelemetry(
            mission_id=enriched_action.get("mission_id", "M_LIVE"),
            action_id=enriched_action.get("action_id", "A_LIVE"),
            current_waypoint=loc_id,
            corridor_name=loc_meta.get("corridor_name", "Lake County Roadway"),
            battery_pct=battery,
            x_coord=float(enriched_action.get("x_coord", 50.0)),
            y_coord=float(enriched_action.get("y_coord", 50.0)),
            payload_status="ALERT" if is_major else "NOMINAL",
            edge_prediction=prediction,
        )
        self.telemetry_history.append(telemetry)

        event_payload = {
            "source": "SPOT_EDGE_PAYLOAD",
            "action_id": telemetry.action_id,
            "mission_id": telemetry.mission_id,
            "waypoint": loc_id,
            "corridor": telemetry.corridor_name,
            "battery_pct": battery,
            "prediction": prediction,
            "cv_features": cv_features,
            "requires_immediate_action": requires_immediate_action,
            "timestamp": telemetry.timestamp,
        }

        # Trigger downstream broadcast if major hazard detected
        if is_major:
            for sub in self.alert_subscribers:
                sub(event_payload)

        return event_payload
