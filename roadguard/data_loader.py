"""
RoadGuard Agent: Data Ingestion & Lake County Topology Enrichment Pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from roadguard.config import DEFAULT_DATA_DIR, LAKE_COUNTY_WAYPOINTS


class SpotDataLoader:
    """
    Loads, joins, enriches, and validates NextEra SPOT Challenge Kit data
    with Lake County, Florida roadway topology and infrastructure metadata.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self._validate_dir()

    def _validate_dir(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"SPOT kit data directory not found: {self.data_dir}"
            )

    def load_raw_tables(self) -> Dict[str, pd.DataFrame]:
        """Loads all CSV tables from the challenge kit."""
        tables = {}
        csv_names = [
            "missions",
            "actions",
            "readings",
            "anomalies",
            "locations",
            "assets",
        ]
        for name in csv_names:
            p = self.data_dir / f"{name}.csv"
            if p.exists():
                tables[name] = pd.read_csv(p)
            else:
                raise FileNotFoundError(f"Missing required table: {p}")
        return tables

    def load_replay_events(self) -> List[Dict]:
        """Loads chronological streaming events from replay.ndjson."""
        p = self.data_dir / "replay.ndjson"
        if not p.exists():
            return []
        events = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events

    def load_validation_benchmarks(self) -> Dict:
        """Loads official validation results JSON."""
        p = self.data_dir / "validation_results.json"
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_unified_dataset(self) -> pd.DataFrame:
        """
        Merges actions, missions, anomalies, readings, assets, and locations,
        derives the major_liability target, and enriches with Lake County GIS context.
        """
        raw = self.load_raw_tables()
        actions = raw["actions"].copy()
        missions = raw["missions"].copy()
        readings = raw["readings"].copy()
        anomalies = raw["anomalies"].copy()
        locations = raw["locations"].copy()
        assets = raw["assets"].copy()

        # 1. Pivot readings by action_id to extract specific measurement types
        # Measurement types: VISUAL_SCORE, LOCALIZATION_CONFIDENCE_PCT, TEMPERATURE_C, SOUND_DB, PRESSURE_PSI
        readings_pivot = readings.pivot_table(
            index="action_id",
            columns="measurement_type",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Rename measurement columns cleanly
        reading_col_map = {
            "VISUAL_SCORE": "reading_visual_score",
            "LOCALIZATION_CONFIDENCE_PCT": "reading_localization_confidence_pct",
            "TEMPERATURE_C": "reading_temperature_c",
            "SOUND_DB": "reading_sound_db",
            "PRESSURE_PSI": "reading_pressure_psi",
        }
        readings_pivot.rename(columns=reading_col_map, inplace=True)

        # Also get reading quality per action
        quality_map = readings.groupby("action_id")["quality"].first().to_dict()
        readings_pivot["reading_quality"] = readings_pivot["action_id"].map(quality_map)

        # 2. Join actions with anomalies
        # Note: Anomalies link on action_id
        anom_subset = anomalies[
            [
                "anomaly_id",
                "action_id",
                "anomaly_type",
                "severity",
                "operator_review_required",
                "description",
            ]
        ].copy()
        anom_subset.rename(
            columns={
                "anomaly_type": "detected_anomaly_type",
                "severity": "anomaly_severity",
                "description": "anomaly_description",
            },
            inplace=True,
        )

        merged = actions.merge(anom_subset, on="action_id", how="left")

        # 3. Join with readings
        merged = merged.merge(readings_pivot, on="action_id", how="left")

        # 4. Join with assets (criticality, asset_type, asset_name)
        assets_subset = assets[
            ["asset_id", "asset_name", "asset_type", "criticality"]
        ].copy()
        assets_subset.rename(
            columns={"criticality": "asset_criticality"}, inplace=True
        )
        merged = merged.merge(assets_subset, on="asset_id", how="left")

        # 5. Join with locations
        loc_subset = locations[
            [
                "location_id",
                "location_name",
                "zone_type",
                "floor",
                "x_coord",
                "y_coord",
                "environment",
            ]
        ].copy()
        loc_subset.rename(
            columns={"location_name": "facility_location_name"}, inplace=True
        )
        merged = merged.merge(loc_subset, on="location_id", how="left")

        # 6. Join with missions (mission duration, start/end battery, scenario, config version)
        missions_subset = missions[
            [
                "mission_id",
                "mission_status",
                "operator_mode",
                "duration_min",
                "start_battery_pct",
                "end_battery_pct",
            ]
        ].copy()
        merged = merged.merge(missions_subset, on="mission_id", how="left")

        # 7. Formulate Ground Truth Binary Target: major_liability (1 or 0)
        # Condition:
        # A. Anomaly severity in CRITICAL or HIGH (which perfectly aligns with operator_review_required == True)
        # OR
        # B. Action failed on a CRITICAL or HIGH criticality asset
        has_critical_high_anomaly = merged["anomaly_severity"].isin(["CRITICAL", "HIGH"])
        is_failed_critical_asset = (merged["status"] == "FAILED") & (
            merged["asset_criticality"].isin(["CRITICAL", "HIGH"])
        )
        merged["major_liability"] = (
            has_critical_high_anomaly | is_failed_critical_asset
        ).astype(int)

        # Also store categorical liability level for granular agent explanations
        def get_liability_tier(row):
            if row["major_liability"] == 1:
                if row["anomaly_severity"] == "CRITICAL" or row["asset_criticality"] == "CRITICAL":
                    return "CRITICAL_LIABILITY"
                return "HIGH_LIABILITY"
            elif pd.notna(row["anomaly_severity"]):
                return "MINOR_WEAR"
            elif row["status"] == "FAILED":
                return "OPERATIONAL_FAILURE"
            return "NOMINAL"

        merged["liability_tier"] = merged.apply(get_liability_tier, axis=1)

        # 8. Enrich with Lake County, Florida GIS & Infrastructure Risk Metadata
        merged["lake_county_corridor"] = merged["location_id"].map(
            lambda loc: LAKE_COUNTY_WAYPOINTS.get(loc, {}).get("corridor_name", "Lake County Roadway")
        )
        merged["lake_county_classification"] = merged["location_id"].map(
            lambda loc: LAKE_COUNTY_WAYPOINTS.get(loc, {}).get("road_classification", "Arterial")
        )
        merged["lake_county_aadt"] = merged["location_id"].map(
            lambda loc: LAKE_COUNTY_WAYPOINTS.get(loc, {}).get("annual_daily_traffic", 15000)
        )
        merged["lake_county_flood_zone"] = merged["location_id"].map(
            lambda loc: LAKE_COUNTY_WAYPOINTS.get(loc, {}).get("flood_zone", "X")
        )
        merged["lake_county_karst_risk"] = merged["location_id"].map(
            lambda loc: LAKE_COUNTY_WAYPOINTS.get(loc, {}).get("karst_subsidence_risk", "Moderate")
        )
        merged["lake_county_primary_risk"] = merged["location_id"].map(
            lambda loc: LAKE_COUNTY_WAYPOINTS.get(loc, {}).get("primary_risk", "Pavement Wear")
        )
        merged["speed_limit_mph"] = merged["location_id"].map(
            lambda loc: LAKE_COUNTY_WAYPOINTS.get(loc, {}).get("speed_limit_mph", 45)
        )

        return merged


def get_dataset(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Convenience helper to retrieve the enriched RoadGuard dataset."""
    loader = SpotDataLoader(data_dir)
    return loader.build_unified_dataset()
