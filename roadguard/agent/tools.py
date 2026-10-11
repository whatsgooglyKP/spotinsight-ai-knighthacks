"""
RoadGuard Agent: Function Calling Tools for Multi-Agent Orchestration.
Exposes geospatial GIS querying, ML classifier inference, multimodal vision verification,
financial risk calculation, and AR HUD packet generation.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

from roadguard.config import DEFAULT_ASSETS_DIR, FINANCIAL_PARAMS, LAKE_COUNTY_WAYPOINTS
from roadguard.financial.impact_quantifier import FinancialImpactQuantifier
from roadguard.hardware.ar_hud_service import SmartGlassesARHUDService
from roadguard.models.classifier import RoadGuardPredictor
from roadguard.models.vision_features import VisionFeatureExtractor


class RoadGuardToolRegistry:
    """
    Central tool executor for Gemini and multi-agent reasoning.
    Provides standard JSON schemas and execution hooks.
    """

    def __init__(self, predictor: Optional[RoadGuardPredictor] = None):
        self.predictor = predictor or RoadGuardPredictor()
        self.vision_extractor = VisionFeatureExtractor()
        self.impact_quantifier = FinancialImpactQuantifier()
        self.ar_service = SmartGlassesARHUDService()

    def get_tool_declarations(self) -> list:
        """Returns tool schema definitions compatible with Gemini function calling."""
        return [
            {
                "name": "geospatial_lake_county_query",
                "description": "Queries Lake County, Florida roadway classification, traffic volume (AADT), karst subsidence risk, and flood zone exposure for an inspection waypoint.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "waypoint_id": {
                            "type": "string",
                            "description": "Waypoint ID, e.g., 'WP01' through 'WP15'.",
                        }
                    },
                    "required": ["waypoint_id"],
                },
            },
            {
                "name": "run_classifier_inference",
                "description": "Executes the supervised learning classifier on SPOT robot inspection telemetry to predict if a fault represents a major liability (1) or non-major wear (0).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_id": {"type": "string"},
                        "action_type": {"type": "string"},
                        "asset_criticality": {"type": "string"},
                        "battery_pct": {"type": "number"},
                        "status": {"type": "string"},
                        "failure_type": {"type": "string"},
                        "detected_anomaly_type": {"type": "string"},
                        "reading_visual_score": {"type": "number"},
                        "reading_temperature_c": {"type": "number"},
                        "reading_sound_db": {"type": "number"},
                        "reading_localization_confidence_pct": {"type": "number"},
                        "reading_pressure_psi": {"type": "number"},
                    },
                    "required": ["location_id", "action_type", "status"],
                },
            },
            {
                "name": "multimodal_vision_verify",
                "description": "Performs multimodal computer vision verification on inspection imagery to detect surface cracks, cavity voids, and thermal hotspots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path_or_name": {
                            "type": "string",
                            "description": "Filename or path of inspection image (e.g., 'pipe_gallery_01.jpg', 'acoustic_valve_01.jpg').",
                        }
                    },
                    "required": ["image_path_or_name"],
                },
            },
            {
                "name": "calculate_financial_actuarial_impact",
                "description": "Calculates municipal tort liability exposure (FL Stat § 768.28), emergency failure cost avoidance, and green bond ESG metrics for a detected fault.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_id": {"type": "string"},
                        "major_liability": {"type": "boolean"},
                        "failure_probability": {"type": "number"},
                    },
                    "required": ["location_id", "major_liability", "failure_probability"],
                },
            },
            {
                "name": "format_smart_glasses_hud",
                "description": "Generates a tactical 3D spatial AR Heads-Up Display packet for smart glasses (Apple Vision Pro, Meta Quest, Vuzix) worn by field technicians.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "waypoint_id": {"type": "string"},
                        "major_liability": {"type": "boolean"},
                        "probability": {"type": "number"},
                        "net_savings_usd": {"type": "number"},
                        "urgency": {"type": "string"},
                    },
                    "required": ["waypoint_id", "major_liability", "probability"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches and executes a tool call by name."""
        if tool_name == "geospatial_lake_county_query":
            wp = args.get("waypoint_id", "WP01").upper()
            meta = LAKE_COUNTY_WAYPOINTS.get(wp)
            if not meta:
                return {"error": f"Unknown waypoint '{wp}'. Valid waypoints: WP01 - WP15."}
            return {"waypoint": wp, **meta}

        elif tool_name == "run_classifier_inference":
            return self.predictor.predict_action(args)

        elif tool_name == "multimodal_vision_verify":
            img_ref = args.get("image_path_or_name", "")
            img_path = Path(img_ref)
            if not img_path.exists():
                # Check within kit sample assets
                alt_path = DEFAULT_ASSETS_DIR / "sample_inspection_images" / img_ref
                if alt_path.exists():
                    img_path = alt_path
                else:
                    return {
                        "status": "IMAGE_NOT_FOUND",
                        "analyzed": False,
                        "fallback_features": self.vision_extractor._default_features(),
                    }

            features = self.vision_extractor.extract_features_from_image(img_path)
            distress = features["cv_overall_distress_score"]
            assessment = (
                "CRITICAL_STRUCTURAL_DEFECT: Extensive edge fracture texture and cavity void detected."
                if distress > 50.0
                else (
                    "MODERATE_SURFACE_WEAR: Hairline weathering patterns observed."
                    if distress > 20.0
                    else "NOMINAL_INTEGRITY: Clean surface profile with no acute void indicators."
                )
            )
            return {
                "image_file": img_path.name,
                "metrics": features,
                "multimodal_verification_assessment": assessment,
            }

        elif tool_name == "calculate_financial_actuarial_impact":
            loc_id = args.get("location_id", "WP01")
            pred = {
                "major_liability_prediction": 1 if args.get("major_liability") else 0,
                "major_liability_probability": float(args.get("failure_probability", 0.5)),
            }
            action_data = {"location_id": loc_id}
            return self.impact_quantifier.quantify_fault_impact(action_data, pred)

        elif tool_name == "format_smart_glasses_hud":
            loc_id = args.get("waypoint_id", "WP01")
            pred = {
                "major_liability_prediction": 1 if args.get("major_liability") else 0,
                "major_liability_probability": float(args.get("probability", 0.5)),
            }
            edge_event = {
                "waypoint": loc_id,
                "corridor": LAKE_COUNTY_WAYPOINTS.get(loc_id, {}).get("corridor_name", "Lake County Roadway"),
                "battery_pct": 74.0,
                "prediction": pred,
            }
            financial_data = {
                "net_capital_cost_avoidance_usd": float(args.get("net_savings_usd", 180800.0)),
                "funding_urgency": args.get("urgency", "TIER 1 - IMMEDIATE CAPITAL DISPATCH"),
            }
            return self.ar_service.format_ar_hud_packet(edge_event, financial_data)

        else:
            return {"error": f"Unknown tool: {tool_name}"}
