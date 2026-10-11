"""
RoadGuard Agent: Financial Risk Actuary & Environmental Impact Quantifier.
Models municipal tort liability exposure, emergency repair cost avoidance,
green bond ESG metrics, and insurance risk pool credits for Lake County, FL.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from roadguard.config import FINANCIAL_PARAMS, LAKE_COUNTY_WAYPOINTS


class FinancialImpactQuantifier:
    """
    Transforms detected roadway faults and major liability predictions
    into actionable municipal finance, insurance underwriting, and ESG capital data.
    """

    def __init__(self, params: Dict[str, float] = None):
        self.params = params or FINANCIAL_PARAMS

    def quantify_fault_impact(self, action_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantifies the financial, safety, and environmental impact of a single inspection fault.
        """
        is_major = prediction_result.get("major_liability_prediction", 0) == 1
        prob = prediction_result.get("major_liability_probability", 0.05)
        location_id = action_data.get("location_id", "WP01")
        loc_meta = LAKE_COUNTY_WAYPOINTS.get(location_id, {})

        aadt = float(loc_meta.get("annual_daily_traffic", 20000))
        speed_mph = float(loc_meta.get("speed_limit_mph", 45))
        corridor = loc_meta.get("corridor_name", "Lake County Roadway")
        karst_risk = loc_meta.get("karst_subsidence_risk", "Moderate")

        # 1. Accident Probability Formulation based on traffic volume, speed, and fault severity
        # Baseline annual accident risk per million vehicle miles: ~1.2
        # Severe structural defect / sinkhole in high speed zone multiplies risk significantly
        severity_multiplier = (1.5 if is_major else 0.2) * (1.5 if karst_risk in ["High", "Severe"] else 1.0)
        exposure_factor = (aadt / 25000.0) * (speed_mph / 45.0)
        annual_accident_prob = float(np.clip(0.015 * exposure_factor * severity_multiplier * (prob + 0.2), 0.005, 0.35))

        # 2. Expected Tort Liability Exposure (FL Stat § 768.28)
        expected_tort_liability = round(annual_accident_prob * self.params["avg_catastrophic_tort_claim"], 2)

        # 3. Emergency vs Proactive Cost Avoidance
        if is_major:
            proactive_cost = self.params["proactive_robotic_repair_cost"]
            emergency_cost = self.params["emergency_road_closure_cost"]
            net_cost_avoidance = round(emergency_cost - proactive_cost, 2)
            emergency_closure_days_avoided = 4.5
        else:
            proactive_cost = 850.0  # Routine surface maintenance
            emergency_cost = 2500.0
            net_cost_avoidance = round(emergency_cost - proactive_cost, 2)
            emergency_closure_days_avoided = 0.5

        # 4. Environmental & Carbon Avoidance Co-Benefits
        # Smoother pavement surface reduces rolling resistance and vehicle idling during lane closures
        # 4.5 closure days avoided = ~18,500 congested vehicle hours avoided
        avoided_fuel_gallons = round((aadt * 0.08 * (emergency_closure_days_avoided / 30.0)) * 0.45, 1)
        avoided_co2_metric_tons = round(avoided_fuel_gallons * 0.008887, 2)
        social_carbon_savings = round(avoided_co2_metric_tons * self.params["carbon_cost_per_ton"], 2)

        # 5. Municipal Capital ROI & Priority Scoring
        # ROI = (Expected Liability Avoided + Direct Cost Avoidance) / Proactive Repair Cost
        total_financial_protection = expected_tort_liability + net_cost_avoidance + social_carbon_savings
        roi_ratio = round(total_financial_protection / proactive_cost, 1)

        # Priority Score (0-100)
        priority_score = float(
            np.clip(
                (prob * 45.0)
                + (min(roi_ratio, 50.0) / 50.0 * 30.0)
                + (min(aadt, 50000.0) / 50000.0 * 25.0),
                0.0,
                100.0,
            )
        )

        return {
            "corridor_name": corridor,
            "location_id": location_id,
            "major_liability": is_major,
            "failure_probability": prob,
            "annual_accident_probability_pct": round(annual_accident_prob * 100, 2),
            "expected_tort_liability_exposure_usd": expected_tort_liability,
            "proactive_repair_cost_usd": proactive_cost,
            "emergency_failure_remediation_cost_usd": emergency_cost,
            "net_capital_cost_avoidance_usd": net_cost_avoidance,
            "total_financial_protection_usd": round(total_financial_protection, 2),
            "roi_ratio": roi_ratio,
            "priority_score": round(priority_score, 1),
            "funding_urgency": (
                "TIER 1 - IMMEDIATE CAPITAL DISPATCH (Within 24h)"
                if priority_score >= 75.0
                else (
                    "TIER 2 - SCHEDULED 30-DAY WORK ORDER"
                    if priority_score >= 45.0
                    else "TIER 3 - ROUTINE MAINTENANCE CYCLE"
                )
            ),
            "environmental_co_benefits": {
                "avoided_co2_metric_tons": avoided_co2_metric_tons,
                "avoided_fuel_gallons": avoided_fuel_gallons,
                "social_carbon_savings_usd": social_carbon_savings,
                "sediment_runoff_avoided_lbs": 320 if is_major else 45,
            },
            "insurance_impact": {
                "risk_rating": "CATASTROPHIC_DEFECT_RISK" if is_major else "STANDARD_OPERATING_RISK",
                "recommended_underwriting_credit_pct": 8.5 if not is_major else 0.0,
                "reserve_allocation_recommended_usd": expected_tort_liability if is_major else 0.0,
            },
        }

    def generate_fleet_portfolio_analysis(self, enriched_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates multiple inspection assessments into an executive municipal portfolio report.
        """
        total_evaluated = len(enriched_predictions)
        major_liabilities = [p for p in enriched_predictions if p["major_liability"]]

        total_tort_avoidance = sum(p["expected_tort_liability_exposure_usd"] for p in major_liabilities)
        total_capital_avoidance = sum(p["net_capital_cost_avoidance_usd"] for p in major_liabilities)
        total_proactive_investment = sum(p["proactive_repair_cost_usd"] for p in major_liabilities)
        total_co2_avoided = sum(p["environmental_co_benefits"]["avoided_co2_metric_tons"] for p in major_liabilities)

        portfolio_roi = (
            round((total_tort_avoidance + total_capital_avoidance) / max(total_proactive_investment, 1.0), 2)
        )

        return {
            "total_inspections_analyzed": total_evaluated,
            "major_liabilities_isolated": len(major_liabilities),
            "minor_cosmetic_defects": total_evaluated - len(major_liabilities),
            "total_proactive_capital_required_usd": round(total_proactive_investment, 2),
            "total_emergency_cost_avoidance_usd": round(total_capital_avoidance, 2),
            "total_tort_liability_exposure_mitigated_usd": round(total_tort_avoidance, 2),
            "total_net_municipal_savings_usd": round(
                total_tort_avoidance + total_capital_avoidance - total_proactive_investment, 2
            ),
            "portfolio_roi_multiple": portfolio_roi,
            "total_emissions_avoidance_co2_metric_tons": round(total_co2_avoided, 2),
            "recommended_capital_appropriation": "LAKE COUNTY INFRASTRUCTURE RESILIENCE FUND - TITLE 34B",
        }
