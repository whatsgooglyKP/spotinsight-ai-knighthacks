"""
RoadGuard Agent: Multi-Agent Orchestration & Gemini Reasoning Brain.
Orchestrates autonomous robotic telemetry ingestion, ML classifier execution,
multimodal vision verification, Lake County geospatial analysis, and financial reporting.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from roadguard.agent.tools import RoadGuardToolRegistry
from roadguard.financial.impact_quantifier import FinancialImpactQuantifier
from roadguard.financial.municipal_report import format_executive_markdown_report
from roadguard.models.classifier import RoadGuardPredictor


class RoadGuardAgent:
    """
    Autonomous multi-agent system combining supervised ML, computer vision,
    geospatial reasoning, and financial actuarial intelligence.
    """

    def __init__(
        self,
        predictor: Optional[RoadGuardPredictor] = None,
        gemini_api_key: Optional[str] = None,
    ):
        self.predictor = predictor or RoadGuardPredictor()
        self.tools = RoadGuardToolRegistry(self.predictor)
        self.impact_quantifier = FinancialImpactQuantifier()
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    def inspect_and_triage_fault(
        self,
        action_telemetry: Dict[str, Any],
        inspection_image: Optional[str] = None,
        explain_cot: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes a complete multi-agent reasoning trajectory on an incoming fault:
        Step 1: SPOT Payload & Safety Agent runs ML classifier inference.
        Step 2: Multimodal Vision Verifier inspects surface distress imagery.
        Step 3: Geospatial Agent evaluates Lake County karst, flood, and traffic exposure.
        Step 4: Actuarial Agent quantifies tort liability and cost avoidance.
        Step 5: AR HUD Agent generates smart-glasses spatial overlay packet.
        Step 6: Synthesizes a formal multi-agent decision & financial recommendation.
        """
        loc_id = action_telemetry.get("location_id", "WP05")
        action_id = action_telemetry.get("action_id", "ACT-INSPECT")

        cot_steps = []

        # Step 1: Tool Call - Run Supervised Classifier
        cot_steps.append({
            "agent": "Civil Engineering & SPOT Safety Agent",
            "action": "Invoking tool 'run_classifier_inference' on robotic telemetry payload.",
        })
        ml_result = self.tools.execute_tool("run_classifier_inference", action_telemetry)
        is_major = ml_result.get("major_liability_prediction", 0) == 1
        prob = ml_result.get("major_liability_probability", 0.5)

        cot_steps.append({
            "agent": "Civil Engineering & SPOT Safety Agent",
            "observation": f"Classifier predicted major_liability={is_major} with probability {prob * 100:.2f}%. Model: {ml_result.get('model_name')}.",
        })

        # Step 2: Tool Call - Multimodal Vision Verification (if image available)
        cv_result = {}
        if inspection_image:
            cot_steps.append({
                "agent": "Multimodal Vision Verification Agent",
                "action": f"Invoking tool 'multimodal_vision_verify' on image '{inspection_image}'.",
            })
            cv_result = self.tools.execute_tool(
                "multimodal_vision_verify", {"image_path_or_name": inspection_image}
            )
            distress_score = cv_result.get("metrics", {}).get("cv_overall_distress_score", 0.0)
            cot_steps.append({
                "agent": "Multimodal Vision Verification Agent",
                "observation": f"Vision analysis completed. Overall distress score: {distress_score}/100. Assessment: {cv_result.get('multimodal_verification_assessment')}",
            })

        # Step 3: Tool Call - Geospatial & Climate Exposure Query
        cot_steps.append({
            "agent": "Geospatial & Climate Risk Agent",
            "action": f"Invoking tool 'geospatial_lake_county_query' for waypoint '{loc_id}'.",
        })
        gis_result = self.tools.execute_tool(
            "geospatial_lake_county_query", {"waypoint_id": loc_id}
        )
        corridor = gis_result.get("corridor_name", "Lake County Roadway")
        karst_risk = gis_result.get("karst_subsidence_risk", "Moderate")
        aadt = gis_result.get("annual_daily_traffic", 20000)

        cot_steps.append({
            "agent": "Geospatial & Climate Risk Agent",
            "observation": f"Corridor: {corridor}. Karst Subsidence Risk: {karst_risk}. AADT: {aadt:,}. Flood Zone: {gis_result.get('flood_zone')}.",
        })

        # Step 4: Tool Call - Actuarial Financial Quantification
        cot_steps.append({
            "agent": "Municipal Finance & Actuarial Agent",
            "action": f"Invoking tool 'calculate_financial_actuarial_impact' for '{loc_id}' (major={is_major}).",
        })
        financial_result = self.tools.execute_tool(
            "calculate_financial_actuarial_impact",
            {
                "location_id": loc_id,
                "major_liability": is_major,
                "failure_probability": prob,
            },
        )

        avoidance = financial_result.get("net_capital_cost_avoidance_usd", 0.0)
        tort_exp = financial_result.get("expected_tort_liability_exposure_usd", 0.0)
        urgency = financial_result.get("funding_urgency", "Standard")
        priority = financial_result.get("priority_score", 50.0)

        cot_steps.append({
            "agent": "Municipal Finance & Actuarial Agent",
            "observation": f"Net Capital Cost Avoidance: ${avoidance:,.2f}. Mitigated Tort Liability: ${tort_exp:,.2f}. Priority Score: {priority}/100 ({urgency}).",
        })

        # Step 5: Tool Call - AR HUD Smart Glasses Formatter
        cot_steps.append({
            "agent": "Hardware & AR Telemetry Agent",
            "action": "Invoking tool 'format_smart_glasses_hud' to prepare heads-up field technician packet.",
        })
        ar_packet = self.tools.execute_tool(
            "format_smart_glasses_hud",
            {
                "waypoint_id": loc_id,
                "major_liability": is_major,
                "probability": prob,
                "net_savings_usd": avoidance,
                "urgency": urgency,
            },
        )
        cot_steps.append({
            "agent": "Hardware & AR Telemetry Agent",
            "observation": f"AR HUD packet generated for Apple Vision Pro / Meta Quest. Reticle: {ar_packet.get('spatial_bounding_box', {}).get('overlay_reticle')}.",
        })

        # Step 6: Synthesis & Actionable Recommendation
        decision = {
            "status": "ANALYSIS_COMPLETE",
            "action_id": action_id,
            "waypoint_id": loc_id,
            "corridor_name": corridor,
            "classification": {
                "major_liability": is_major,
                "confidence_score": prob,
                "rating": "MAJOR LIABILITY (CRITICAL SAFETY THREAT)" if is_major else "NON-MAJOR (ROUTINE SURFACE WEAR)",
                "requires_immediate_operator_review": is_major,
            },
            "geospatial_context": gis_result,
            "computer_vision_verification": cv_result,
            "financial_actuarial_metrics": financial_result,
            "smart_glasses_ar_hud": ar_packet,
            "executive_recommendation": (
                f"URGENT CAPITAL INTERVENTION: The defect at {corridor} represents an acute major liability. "
                f"Immediate dispatch of a robotic repair crew avoids ${avoidance:,.2f} in emergency reconstruction and mitigates ${tort_exp:,.2f} in statutory municipal tort liability. "
                f"Field technicians are receiving AR hazard bounding boxes via smart glasses."
                if is_major
                else f"ROUTINE MAINTENANCE: Defect at {corridor} is classified as non-major surface wear. "
                f"Logged into scheduled 30-day resurfacing rotation with low legal liability."
            ),
        }

        if explain_cot:
            decision["chain_of_thought_trajectory"] = cot_steps

        return decision

    def generate_county_wide_audit(self, actions_df) -> Dict[str, Any]:
        """
        Runs full county-wide audit on all actions in the dataset,
        generates ranked capital schedules, portfolio ROI, and executive reports.
        """
        records = actions_df.to_dict(orient="records")
        results = []
        for rec in records:
            pred = self.predictor.predict_action(rec)
            impact = self.impact_quantifier.quantify_fault_impact(rec, pred)
            combined = {**rec, **pred, **impact}
            results.append(combined)

        portfolio = self.impact_quantifier.generate_fleet_portfolio_analysis(results)
        markdown_report = format_executive_markdown_report(portfolio, results)

        return {
            "portfolio_summary": portfolio,
            "detailed_action_evaluations": results,
            "executive_markdown_report": markdown_report,
        }
