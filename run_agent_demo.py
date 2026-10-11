"""
run_agent_demo.py - End-to-End Demonstration of RoadGuard Agent

Executes multi-agent inference on representative SPOT robot inspection events
in Lake County, Florida:
  Event A: Critical Karst Roadway Subsurface Void (Mount Dora - WP05) -> MAJOR LIABILITY (1)
  Event B: Minor Surface Aggregate Wear (Clermont Causeway - WP01)   -> NON-MAJOR (0)
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roadguard.models.classifier import RoadGuardPredictor
from roadguard.agent.orchestrator import RoadGuardAgent
from roadguard.hardware.spot_payload_service import SpotPayloadEdgeService
from roadguard.hardware.ar_hud_service import SmartGlassesARHUDService
from roadguard.config import SAVED_MODEL_PATH

def main():
    print("=" * 80)
    print(" RoadGuard Agent: End-to-End Multi-Agent System Demonstration")
    print(" Knight Hacks IX (UCF) - AI Boundary / Financial Services / Most Prestigious")
    print(" Scoped to Lake County, FL Infrastructure & NextEra SPOT Autonomous Patrol")
    print("=" * 80)

    # Initialize predictor and orchestrator
    print("\n[INIT] Initializing Trained Supervised Classifier & Multi-Agent Orchestrator...")
    predictor = RoadGuardPredictor(SAVED_MODEL_PATH)
    agent = RoadGuardAgent(predictor=predictor)
    spot_service = SpotPayloadEdgeService(predictor=predictor)

    # Define Test Event A: Major Structural Karst Failure at Mount Dora (WP05)
    event_a_payload = {
        "mission_id": "MISSION-LAKE-092",
        "action_id": "ACT-108",
        "location_id": "WP05",
        "asset_id": "ASSET-CV-KARST-05",
        "action_type": "INSPECT_INFRASTRUCTURE",
        "status": "FAILED",
        "duration_sec": 48.5,
        "battery_pct": 74.2,
        "has_anomaly": 1,
        "anomaly_id": "ANOM-701",
        "failure_type": "STRUCTURAL_VOID",
        "severity": "CRITICAL",
        "operator_review_required": True,
        "reading_visual_score": 12.0,            # Severe visual degradation
        "reading_localization_pct": 68.5,        # Severe localization drop / terrain shift
        "reading_temperature_c": 39.8,
        "reading_sound_db": 88.4,                # Subsurface acoustic resonance
        "reading_pressure_psi": 29.5,
        "asset_criticality": "CRITICAL",
        "zone_type": "ACCESS",
        "x_coord": 45.2,
        "y_coord": 88.1,
    }

    # Define Test Event B: Minor Aggregate Wear at Clermont Causeway (WP01)
    event_b_payload = {
        "mission_id": "MISSION-LAKE-093",
        "action_id": "ACT-109",
        "location_id": "WP01",
        "asset_id": "ASSET-CV-WEAR-01",
        "action_type": "INSPECT_INFRASTRUCTURE",
        "status": "SUCCESS",
        "duration_sec": 12.0,
        "battery_pct": 86.0,
        "has_anomaly": 0,
        "anomaly_id": None,
        "failure_type": "NONE",
        "severity": "LOW",
        "operator_review_required": False,
        "reading_visual_score": 88.0,
        "reading_localization_pct": 98.2,
        "reading_temperature_c": 28.5,
        "reading_sound_db": 54.0,
        "reading_pressure_psi": 32.0,
        "asset_criticality": "LOW",
        "zone_type": "ACCESS",
        "x_coord": 22.0,
        "y_coord": 31.4,
    }

    scenarios = [
        ("SCENARIO 1: CRITICAL KARST FAILURE (MOUNT DORA - WP05)", event_a_payload),
        ("SCENARIO 2: ROUTINE SURFACE WEAR (CLERMONT CAUSEWAY - WP01)", event_b_payload)
    ]

    for title, payload in scenarios:
        print("\n" + "#" * 80)
        print(f" >>> {title}")
        print("#" * 80)

        # 1. SPOT Robot Edge Perception Simulation
        print("\n[1. SPOT EDGE PAYLOAD TELEMETRY]")
        spot_telemetry = spot_service.process_incoming_action(payload)
        edge_pred = spot_telemetry["prediction"]
        print(f"  Robot Status: {spot_telemetry['source']} | Waypoint: {spot_telemetry['waypoint']} ({spot_telemetry['corridor']})")
        print(f"  Battery: {spot_telemetry['battery_pct']}% | Alert Flag: {spot_telemetry['requires_immediate_action']}")
        print(f"  Edge Model P(Major Liability): {edge_pred['major_liability_probability']:.3f} -> Predicted Class: {edge_pred['major_liability_prediction']}")

        # 2. Multi-Agent Orchestration & Reasoning
        print("\n[2. MULTI-AGENT ORCHESTRATION TRAJECTORY]")
        agent_result = agent.inspect_and_triage_fault(payload)

        # Print Chain of Thought trajectory
        for step in agent_result.get("chain_of_thought_trajectory", []):
            agent_role = step.get("agent", "Agent")
            if "action" in step:
                print(f"  [{agent_role}] -> ACTION: {step['action']}")
            if "observation" in step:
                print(f"  [{agent_role}] -> OBS: {step['observation']}")

        # 3. Financial Actuarial Quantification
        fin = agent_result["financial_actuarial_metrics"]
        print("\n[3. FINANCIAL ACTUARIAL AUDIT - LAKE COUNTY, FL MUNICIPAL SERVICES]")
        print(f"  Target Classification        : {agent_result['classification']['rating']}")
        print(f"  Supervised ML Confidence     : {agent_result['classification']['confidence_score']:.1%}")
        print(f"  Requires Operator Review     : {agent_result['classification']['requires_immediate_operator_review']}")
        print(f"  Priority Score / Urgency     : {fin.get('priority_score')}/100 ({fin.get('funding_urgency')})")
        print(f"  Net Capital Cost Avoidance   : ${fin.get('net_capital_cost_avoidance_usd', 0.0):,.2f}")
        print(f"  Expected Tort Liability Prot : ${fin.get('expected_tort_liability_exposure_usd', 0.0):,.2f}")
        print(f"  Proactive Robotic Repair Est : ${fin.get('proactive_repair_cost_usd', 0.0):,.2f}")
        print(f"  Emergency Reconstruction Est : ${fin.get('emergency_reconstruction_cost_usd', 0.0):,.2f}")
        print(f"  Municipal ROI Multiple       : {fin.get('municipal_roi_multiple', 1.0):.2f}x")
        print(f"  Annual Insurance Premium Cred: ${fin.get('annual_insurance_premium_credit_usd', 0.0):,.2f}")
        print(f"  Green Bond CO2 Abatement     : {fin.get('green_bond_metrics', {}).get('avoided_congestion_co2_kg', 0.0):.1f} kg CO2e")

        # 4. Smart Glasses AR HUD Frame Output
        print("\n[4. SMART GLASSES AR HUD FRAME TELEMETRY]")
        ar_frame = agent_result["smart_glasses_ar_hud"]
        hud_display = ar_frame.get("hud_display", {})
        spatial_box = ar_frame.get("spatial_bounding_box", {})
        haptics = ar_frame.get("wearable_haptics", {})
        print(f"  AR Device Target : {', '.join(ar_frame.get('device_target', []))}")
        print(f"  HUD Theme Color  : {hud_display.get('color_theme')} | Banner: {hud_display.get('banner_title')}")
        print(f"  AR Status Badge  : {hud_display.get('status_badge')}")
        print(f"  Reticle Overlay  : {spatial_box.get('overlay_reticle')} at depth {spatial_box.get('depth_meters')}m (Anchor: {spatial_box.get('world_anchor')})")
        print(f"  Wearable Haptics : {haptics.get('pattern')} | Audio: {haptics.get('audio_cue')} | Vibration: {haptics.get('smartwatch_vibration')}")
        print(f"  Voice Commands   : {ar_frame.get('field_operator_voice_command_options')}")

        # 5. Executive Synthesis
        print("\n[5. EXECUTIVE SYNTHESIS FOR MUNICIPAL LEADERSHIP]")
        print(f"  {agent_result['executive_recommendation']}")

    print("\n" + "=" * 80)
    print(" End-to-End Multi-Agent Demonstration Completed Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
