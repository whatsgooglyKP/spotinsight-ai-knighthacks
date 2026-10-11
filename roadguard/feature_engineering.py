"""
RoadGuard Agent: Machine Learning Feature Engineering Pipeline.
Strictly adheres to /ml-best-practices:
- Strict Featurization Ordering: Fit transformers ONLY on train split, transform test split independently.
- Explicit Handling of Missing & Null Values with contextual domain reasoning.
- Clear separation of numerical scaling, ordinal encoding, and one-hot encoding.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RoadGuardFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant feature engineering transformer for RoadGuard Agent.
    Transforms raw multi-modal SPOT + drone inspection telemetry into ML features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.medians_: Dict[str, float] = {}

    def _extract_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives computer vision, thermal, acoustic, and spatial risk indicators."""
        data = df.copy()

        # Normalize column aliases if passed from disparate sensors
        alias_map = {
            "visual_score": "reading_visual_score",
            "reading_visual": "reading_visual_score",
            "localization_pct": "reading_localization_confidence_pct",
            "reading_localization_pct": "reading_localization_confidence_pct",
            "temperature_c": "reading_temperature_c",
            "sound_db": "reading_sound_db",
            "pressure_psi": "reading_pressure_psi",
            "corridor": "lake_county_corridor",
            "corridor_name": "lake_county_corridor",
            "karst_risk": "lake_county_karst_risk",
            "flood_zone": "lake_county_flood_zone",
            "aadt": "lake_county_aadt",
        }
        for alias, target in alias_map.items():
            if alias in data.columns and target not in data.columns:
                data[target] = data[alias]

        # Ensure all expected categorical and numerical columns exist with robust fallbacks
        col_defaults = {
            "action_type": "INSPECT_INFRASTRUCTURE",
            "failure_type": "NONE",
            "detected_anomaly_type": "NONE",
            "anomaly_id": None,
            "status": "SUCCESS",
            "asset_criticality": "LOW",
            "lake_county_karst_risk": "Moderate",
            "environment": "OUTDOOR",
            "zone_type": "ACCESS",
            "lake_county_flood_zone": "X",
            "scenario": "DEFAULT",
            "configuration_version": "v1.0",
            "reading_visual_score": np.nan,
            "reading_temperature_c": np.nan,
            "reading_sound_db": np.nan,
            "reading_localization_confidence_pct": np.nan,
            "reading_pressure_psi": np.nan,
            "battery_pct": 80.0,
            "duration_sec": 30.0,
            "sequence": 1.0,
            "x_coord": 50.0,
            "y_coord": 50.0,
            "lake_county_aadt": 20000.0,
            "speed_limit_mph": 45.0,
        }
        for col, default_val in col_defaults.items():
            if col not in data.columns:
                data[col] = default_val

        # 1. Fill missing failure types and anomalies contextually
        data["failure_type_clean"] = data["failure_type"].fillna("NONE").astype(str)
        data["anomaly_type_clean"] = data["detected_anomaly_type"].fillna("NONE").astype(str)
        data["has_anomaly"] = data["anomaly_id"].notna().astype(float)
        data["is_action_failed"] = (data["status"] == "FAILED").astype(float)

        # 2. Ordinal Criticality Mapping (Domain-aware)
        criticality_map = {"CRITICAL": 3.0, "HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.0}
        data["asset_criticality_num"] = (
            data["asset_criticality"].map(criticality_map).fillna(0.0)
        )

        karst_map = {"Severe": 3.0, "High": 2.0, "Moderate": 1.0, "Low": 0.0}
        data["karst_risk_num"] = (
            data["lake_county_karst_risk"].map(karst_map).fillna(1.0)
        )

        # 3. Sensor & Computer Vision Reading Features
        # Compute indicator flags for reading modality presence
        data["has_visual_reading"] = data["reading_visual_score"].notna().astype(float)
        data["has_thermal_reading"] = data["reading_temperature_c"].notna().astype(float)
        data["has_acoustic_reading"] = data["reading_sound_db"].notna().astype(float)
        data["has_localization_reading"] = (
            data["reading_localization_confidence_pct"].notna().astype(float)
        )
        data["has_pressure_reading"] = data["reading_pressure_psi"].notna().astype(float)

        # Impute missing values for numeric reading columns using train-fitted medians (or current if fitting)
        reading_defaults = {
            "reading_visual_score": 85.0,
            "reading_temperature_c": 64.0,
            "reading_sound_db": 71.0,
            "reading_localization_confidence_pct": 82.0,
            "reading_pressure_psi": 110.0,
            "battery_pct": 55.0,
            "duration_sec": 30.0,
            "sequence": 5.0,
            "x_coord": 50.0,
            "y_coord": 50.0,
            "lake_county_aadt": 20000.0,
            "speed_limit_mph": 45.0,
        }

        for col, default_val in reading_defaults.items():
            if col in data.columns:
                val = self.medians_.get(col, default_val)
                data[col + "_imputed"] = data[col].fillna(val)

        # 4. Computer Vision & Physical Distress Signals
        # Lower visual inspection score indicates higher surface distress / cracking
        data["cv_surface_defect_ratio"] = np.clip(
            (100.0 - data["reading_visual_score_imputed"]) / 100.0, 0.0, 1.0
        )
        # Elevated temperature deviation (thermal hotspot on pavement or pump)
        data["thermal_heat_elevation"] = np.maximum(
            0.0, data["reading_temperature_c_imputed"] - 65.0
        )
        # Elevated acoustic vibration index (structural rattle / void resonance)
        data["acoustic_vibration_index"] = np.maximum(
            0.0, data["reading_sound_db_imputed"] - 70.0
        )
        # Localization uncertainty (LiDAR drift indicating topological subsidence or occlusion)
        data["lidar_localization_drift"] = np.clip(
            (100.0 - data["reading_localization_confidence_pct_imputed"]) / 100.0,
            0.0,
            1.0,
        )
        # Traffic exposure risk index: (AADT / 10,000) * speed_limit
        data["corridor_kinetic_hazard_index"] = (
            data["lake_county_aadt_imputed"] / 10000.0
        ) * (data["speed_limit_mph_imputed"] / 45.0)

        # Battery depletion stress factor (SPOT operational fatigue)
        data["battery_depletion_stress"] = np.clip(
            (50.0 - data["battery_pct_imputed"]) / 50.0, 0.0, 1.0
        )

        return data

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the preprocessing pipeline ONLY on the training split."""
        df_domain = self._extract_domain_features(X)

        # Calculate and store medians for imputation from training set
        numeric_candidates = [
            "reading_visual_score",
            "reading_temperature_c",
            "reading_sound_db",
            "reading_localization_confidence_pct",
            "reading_pressure_psi",
            "battery_pct",
            "duration_sec",
            "sequence",
            "x_coord",
            "y_coord",
            "lake_county_aadt",
            "speed_limit_mph",
        ]
        for col in numeric_candidates:
            if col in X.columns:
                self.medians_[col] = float(X[col].median(skipna=True))

        # Re-derive domain features with true train medians
        df_domain = self._extract_domain_features(X)

        self.numerical_cols = [
            "battery_pct_imputed",
            "duration_sec_imputed",
            "sequence_imputed",
            "x_coord_imputed",
            "y_coord_imputed",
            "lake_county_aadt_imputed",
            "speed_limit_mph_imputed",
            "asset_criticality_num",
            "karst_risk_num",
            "has_anomaly",
            "is_action_failed",
            "has_visual_reading",
            "has_thermal_reading",
            "has_acoustic_reading",
            "has_localization_reading",
            "has_pressure_reading",
            "cv_surface_defect_ratio",
            "thermal_heat_elevation",
            "acoustic_vibration_index",
            "lidar_localization_drift",
            "corridor_kinetic_hazard_index",
            "battery_depletion_stress",
        ]

        self.categorical_cols = [
            "action_type",
            "failure_type_clean",
            "environment",
            "zone_type",
            "lake_county_flood_zone",
            "scenario",
            "configuration_version",
        ]

        # Fit Scaler on Numerical Features
        self.scaler.fit(df_domain[self.numerical_cols])

        # Fit Encoder on Categorical Features
        self.encoder.fit(df_domain[self.categorical_cols])

        encoded_feature_names = list(
            self.encoder.get_feature_names_out(self.categorical_cols)
        )
        self.feature_names_ = self.numerical_cols + encoded_feature_names
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input data using the fitted training pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before transform.")

        df_domain = self._extract_domain_features(X)

        X_num = self.scaler.transform(df_domain[self.numerical_cols])
        X_cat = self.encoder.transform(df_domain[self.categorical_cols])

        return np.hstack([X_num, X_cat])

    def get_feature_names(self) -> List[str]:
        return self.feature_names_
