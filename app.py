"""
app.py - RoadGuard Agent Interactive Hackathon Judge Dashboard (FastAPI)
Knight Hacks IX (UCF) - AI Boundary / Financial Services / Most Prestigious Tracks
Scoped to Lake County, FL & Boston Dynamics SPOT Robot Inspection Kit
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from roadguard.config import LAKE_COUNTY_WAYPOINTS, SAVED_MODEL_PATH
from roadguard.data_loader import get_dataset
from roadguard.models.classifier import RoadGuardPredictor
from roadguard.agent.orchestrator import RoadGuardAgent
from roadguard.hardware.spot_payload_service import SpotPayloadEdgeService
from roadguard.hardware.ar_hud_service import SmartGlassesARHUDService
from roadguard.financial.impact_quantifier import FinancialImpactQuantifier
from roadguard.hardware.replay_streamer import ReplayStreamer

app = FastAPI(
    title="RoadGuard Agent API & Demo Dashboard",
    description="Multi-Agent Infrastructure Triage, Supervised ML Classifier & Municipal Financial Actuary",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
predictor = RoadGuardPredictor(SAVED_MODEL_PATH)
agent = RoadGuardAgent(predictor=predictor)
spot_service = SpotPayloadEdgeService(predictor=predictor)
ar_service = SmartGlassesARHUDService()
impact_quantifier = FinancialImpactQuantifier()
replay_streamer = ReplayStreamer()

class TriageRequest(BaseModel):
    waypoint_id: str = "WP05"
    action_type: str = "INSPECT_INFRASTRUCTURE"
    status: str = "FAILED"
    visual_score: float = 18.0
    localization_pct: float = 65.0
    temperature_c: float = 41.5
    sound_db: float = 87.0
    pressure_psi: float = 28.0
    battery_pct: float = 72.0
    has_anomaly: int = 1
    failure_type: str = "STRUCTURAL_VOID"
    severity: str = "CRITICAL"
    asset_criticality: str = "CRITICAL"
    image_url: Optional[str] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "ONLINE",
        "system": "RoadGuard Agent Multi-Agent AI",
        "jurisdiction": "Lake County, Florida",
        "model_loaded": predictor.model_name,
        "supported_tracks": [
            "Most Prestigious Track (Spectacular System)",
            "AI Boundary Track (Supervised ML + CV + LLM)",
            "Financial Services Track (Actuarial Workflow Automation)",
            "Hardware/Embedded Track (SPOT Payload & AR Glasses)"
        ]
    }

@app.get("/api/waypoints")
def get_waypoints():
    return LAKE_COUNTY_WAYPOINTS

@app.post("/api/triage")
def triage_fault(req: TriageRequest):
    payload = {
        "mission_id": "MISSION-LIVE-PATROL",
        "action_id": f"ACT-{req.waypoint_id}-LIVE",
        "location_id": req.waypoint_id,
        "action_type": req.action_type,
        "status": req.status,
        "duration_sec": 35.0,
        "battery_pct": req.battery_pct,
        "has_anomaly": req.has_anomaly,
        "anomaly_id": "ANOM-LIVE-01" if req.has_anomaly else None,
        "failure_type": req.failure_type,
        "severity": req.severity,
        "operator_review_required": req.severity in ["CRITICAL", "HIGH"],
        "reading_visual_score": req.visual_score,
        "reading_localization_confidence_pct": req.localization_pct,
        "reading_temperature_c": req.temperature_c,
        "reading_sound_db": req.sound_db,
        "reading_pressure_psi": req.pressure_psi,
        "asset_criticality": req.asset_criticality,
        "zone_type": LAKE_COUNTY_WAYPOINTS.get(req.waypoint_id, {}).get("zone_type", "ACCESS"),
        "x_coord": 45.0,
        "y_coord": 60.0,
    }

    # 1. Edge robot payload processing
    edge_event = spot_service.process_incoming_action(payload)

    # 2. Multi-agent triage & reasoning trajectory
    triage_result = agent.inspect_and_triage_fault(payload)

    return {
        "edge_telemetry": edge_event,
        "triage_decision": triage_result
    }

@app.get("/api/county-audit")
def county_audit():
    actions_df = get_dataset()
    return agent.generate_county_wide_audit(actions_df)

@app.get("/api/replay-sample")
def get_replay_sample(limit: int = 15):
    return {"events": replay_streamer.get_sample_events(limit=limit)}

@app.get("/", response_class=HTMLResponse)
def index_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RoadGuard Agent - Knight Hacks IX Hackathon Submission</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body { background-color: #0B0F19; color: #E2E8F0; font-family: ui-sans-serif, system-ui, sans-serif; }
    .neon-border-red { box-shadow: 0 0 15px rgba(239, 68, 68, 0.4); border-color: #EF4444; }
    .neon-border-green { box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); border-color: #10B981; }
    .hud-reticle { border: 2px dashed #EF4444; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
  </style>
</head>
<body class="min-h-screen p-6">
  <!-- Header -->
  <header class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center pb-6 border-b border-gray-800">
    <div class="flex items-center space-x-3">
      <div class="p-3 bg-red-600/20 border border-red-500 rounded-xl">
        <i class="fa-solid fa-robot text-red-500 text-2xl"></i>
      </div>
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          RoadGuard Agent <span class="text-xs uppercase px-2 py-0.5 rounded bg-blue-900/60 text-blue-400 border border-blue-700">Knight Hacks IX</span>
        </h1>
        <p class="text-xs text-gray-400">Autonomous SPOT Robot Triage &bull; Supervised ML &bull; Municipal Financial Actuary &bull; Lake County, FL</p>
      </div>
    </div>
    <div class="flex items-center gap-2 mt-4 md:mt-0">
      <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-950 text-emerald-400 border border-emerald-800">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span> SPOT Robot Edge Online
      </span>
      <span class="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium bg-purple-950 text-purple-400 border border-purple-800">
        <i class="fa-solid fa-glasses"></i> AR HUD V2 Ready
      </span>
    </div>
  </header>

  <!-- Track Target Badges -->
  <div class="max-w-7xl mx-auto my-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
    <div class="bg-gray-900/80 p-2.5 rounded-lg border border-yellow-500/30 flex items-center gap-2">
      <i class="fa-solid fa-trophy text-yellow-500 text-lg"></i>
      <div><span class="text-yellow-400 font-bold block">Most Prestigious</span><span class="text-gray-400">Unique multi-agent showstopper</span></div>
    </div>
    <div class="bg-gray-900/80 p-2.5 rounded-lg border border-cyan-500/30 flex items-center gap-2">
      <i class="fa-solid fa-brain text-cyan-400 text-lg"></i>
      <div><span class="text-cyan-400 font-bold block">AI Boundary</span><span class="text-gray-400">Real ML (PR-AUC 0.59) + CV + Gemini</span></div>
    </div>
    <div class="bg-gray-900/80 p-2.5 rounded-lg border border-emerald-500/30 flex items-center gap-2">
      <i class="fa-solid fa-sack-dollar text-emerald-400 text-lg"></i>
      <div><span class="text-emerald-400 font-bold block">Financial Services</span><span class="text-gray-400">FL § 768.28 Tort & Bond Underwriting</span></div>
    </div>
    <div class="bg-gray-900/80 p-2.5 rounded-lg border border-purple-500/30 flex items-center gap-2">
      <i class="fa-solid fa-microchip text-purple-400 text-lg"></i>
      <div><span class="text-purple-400 font-bold block">Hardware / SPOT</span><span class="text-gray-400">Payload Edge Daemon + AR Glasses</span></div>
    </div>
  </div>

  <!-- Main Grid -->
  <main class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6 mt-4">
    <!-- Left Column: Controls & Waypoints (4 cols) -->
    <div class="lg:col-span-4 space-y-4">
      <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-5 shadow-xl">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-gray-400 mb-3 flex items-center justify-between">
          <span><i class="fa-solid fa-satellite-dish mr-2 text-blue-400"></i>Mission Waypoint Patrol</span>
          <span class="text-xs text-blue-400 font-normal">NextEra Kit (Lake County)</span>
        </h2>
        
        <label class="block text-xs text-gray-300 mb-1 font-medium">Select Inspection Corridor:</label>
        <select id="waypointSelect" class="w-full bg-gray-800 border border-gray-700 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-blue-500 mb-4" onchange="loadPreset()">
          <option value="WP05" selected>WP05: Mount Dora Karst Pipe Gallery (CRITICAL VOID)</option>
          <option value="WP01">WP01: US-27 South Lake Logistics Corridor (COSMETIC WEAR)</option>
          <option value="WP02">WP02: SR-19 Lake Harris Causeway & Bridge (HIGH SURGE)</option>
          <option value="WP06">WP06: Leesburg Turnpike Overpass (FATIGUE CRACKING)</option>
          <option value="WP10">WP10: Green Mountain Scenic Byway (SUBSIDENCE HAZARD)</option>
        </select>

        <div class="space-y-3 text-xs">
          <div>
            <div class="flex justify-between text-gray-400 mb-1">
              <span>Visual Integrity (CV Proxy)</span>
              <span id="visVal" class="text-white font-mono">18%</span>
            </div>
            <input type="range" id="visual_score" min="5" max="100" value="18" class="w-full accent-red-500" oninput="updateVal('visVal', this.value + '%')">
          </div>
          <div>
            <div class="flex justify-between text-gray-400 mb-1">
              <span>LiDAR Localization Confidence</span>
              <span id="locVal" class="text-white font-mono">65%</span>
            </div>
            <input type="range" id="localization_pct" min="40" max="100" value="65" class="w-full accent-blue-500" oninput="updateVal('locVal', this.value + '%')">
          </div>
          <div>
            <div class="flex justify-between text-gray-400 mb-1">
              <span>Acoustic Void Resonance (dB)</span>
              <span id="sndVal" class="text-white font-mono">87 dB</span>
            </div>
            <input type="range" id="sound_db" min="45" max="105" value="87" class="w-full accent-amber-500" oninput="updateVal('sndVal', this.value + ' dB')">
          </div>
        </div>

        <button onclick="runTriage()" id="triageBtn" class="w-full mt-5 bg-gradient-to-r from-red-600 to-rose-700 hover:from-red-500 hover:to-rose-600 text-white font-bold py-3 px-4 rounded-lg flex items-center justify-center gap-2 shadow-lg transition">
          <i class="fa-solid fa-play"></i> Trigger Autonomous Multi-Agent Triage
        </button>
      </div>

      <!-- Live SPOT Robot Status Card -->
      <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2 flex items-center justify-between">
          <span><i class="fa-solid fa-battery-half text-emerald-400 mr-1.5"></i>Boston Dynamics SPOT Telemetry</span>
          <span class="text-xs text-gray-500 font-mono">Jetson Orin Edge</span>
        </h3>
        <div class="grid grid-cols-2 gap-2 text-xs">
          <div class="bg-gray-800/60 p-2 rounded border border-gray-700/60">
            <span class="text-gray-400 block">Battery Level</span>
            <span class="text-emerald-400 font-bold font-mono text-sm" id="batteryReadout">74.2%</span>
          </div>
          <div class="bg-gray-800/60 p-2 rounded border border-gray-700/60">
            <span class="text-gray-400 block">Edge Classifier</span>
            <span class="text-white font-bold font-mono text-sm">GradBoost v1</span>
          </div>
          <div class="bg-gray-800/60 p-2 rounded border border-gray-700/60">
            <span class="text-gray-400 block">Karst Subsidence</span>
            <span class="text-amber-400 font-semibold" id="karstReadout">Severe Risk</span>
          </div>
          <div class="bg-gray-800/60 p-2 rounded border border-gray-700/60">
            <span class="text-gray-400 block">Traffic (AADT)</span>
            <span class="text-white font-mono" id="aadtReadout">19,500</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Center/Right Column: Live Results & Multi-Agent HUD (8 cols) -->
    <div class="lg:col-span-8 space-y-4">
      <!-- Top Classification & Financial Banner -->
      <div id="resultBanner" class="bg-gray-900/90 border-2 neon-border-red rounded-xl p-5 transition-all">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div class="flex items-center gap-2 mb-1">
              <span id="liabilityBadge" class="px-3 py-1 rounded text-xs font-extrabold uppercase bg-red-600 text-white tracking-wide">
                MAJOR LIABILITY (CLASS 1)
              </span>
              <span id="probBadge" class="text-xs font-mono text-gray-300">ML Confidence: 98.5%</span>
            </div>
            <h3 id="corridorHeading" class="text-lg font-bold text-white">Mount Dora Karst Pipe Gallery / Subgrade Void Zone</h3>
            <p id="recommendationText" class="text-xs text-gray-300 mt-1 max-w-2xl">
              URGENT CAPITAL INTERVENTION: Structural void detected in active karst sinkhole zone. Immediate repair prevents highway collapse and statutory municipal tort lawsuits.
            </p>
          </div>
          <div class="text-right bg-gray-800/80 p-3 rounded-lg border border-gray-700 min-w-[200px]">
            <span class="text-xs text-emerald-400 font-semibold uppercase block">Net Cost Avoidance</span>
            <span id="costAvoidanceVal" class="text-2xl font-black text-emerald-400 font-mono">$180,800</span>
            <span id="roiVal" class="text-xs text-gray-400 block mt-0.5">Municipal ROI: 43.0x</span>
          </div>
        </div>
      </div>

      <!-- Financial Services Actuarial Workflow Matrix -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4">
          <div class="flex items-center justify-between text-xs text-gray-400 mb-1">
            <span>FL Stat § 768.28 Tort Mitigation</span>
            <i class="fa-solid fa-scale-balanced text-blue-400"></i>
          </div>
          <div id="tortVal" class="text-xl font-bold text-white font-mono">$5,659.90</div>
          <p class="text-[11px] text-gray-400 mt-1">Expected municipal liability protection per incident</p>
        </div>

        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4">
          <div class="flex items-center justify-between text-xs text-gray-400 mb-1">
            <span>Insurance Risk Pool Credit</span>
            <i class="fa-solid fa-shield-halved text-purple-400"></i>
          </div>
          <div id="insVal" class="text-xl font-bold text-purple-300 font-mono">$2,400.00 / yr</div>
          <p class="text-[11px] text-gray-400 mt-1">Underwriter discount for continuous robotic audit</p>
        </div>

        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4">
          <div class="flex items-center justify-between text-xs text-gray-400 mb-1">
            <span>Climate / Green Bond CO2</span>
            <i class="fa-solid fa-leaf text-emerald-400"></i>
          </div>
          <div id="co2Val" class="text-xl font-bold text-emerald-400 font-mono">1,420 kg CO2e</div>
          <p class="text-[11px] text-gray-400 mt-1">Emissions avoided by preventing detour congestion</p>
        </div>
      </div>

      <!-- Split: Smart Glasses AR Overlay Preview + Multi-Agent Chain-of-Thought -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- AR Glasses HUD Simulation -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-2">
              <h4 class="text-xs font-semibold uppercase text-purple-400 flex items-center gap-1.5">
                <i class="fa-solid fa-glasses"></i> Apple Vision Pro / Meta Quest 3 AR HUD
              </h4>
              <span class="text-[10px] bg-purple-900/50 text-purple-300 px-2 py-0.5 rounded border border-purple-700">Spatial Depth: 3.8m</span>
            </div>
            
            <!-- Simulated AR Viewport -->
            <div class="relative bg-black/60 rounded-lg h-44 border border-gray-800 overflow-hidden flex flex-col justify-between p-3">
              <div class="flex justify-between text-[11px] font-mono text-red-400">
                <span id="arBannerText">ROADGUARD HAZARD DETECTION</span>
                <span id="arBadgeText">[MAJOR LIABILITY: HIGH RISK]</span>
              </div>

              <!-- Reticle Target in Center -->
              <div class="self-center flex flex-col items-center">
                <div id="reticleBox" class="hud-reticle w-24 h-16 rounded flex items-center justify-center">
                  <i id="reticleIcon" class="fa-solid fa-triangle-exclamation text-red-500 text-2xl"></i>
                </div>
                <span id="reticleLabel" class="text-[10px] text-red-400 font-mono mt-1">ANCHOR: WP05_SURFACE</span>
              </div>

              <div class="flex justify-between text-[10px] font-mono text-gray-400 border-t border-gray-800 pt-1">
                <span>Haptic: DOUBLE_BURST</span>
                <span>Audio: hazard_alert.wav</span>
                <span>Voice: [Confirm | Drone]</span>
              </div>
            </div>
          </div>
          <p class="text-[11px] text-gray-400 mt-2">Field technicians receive real-time spatial boundary geometry and voice dispatch on industrial AR smart glasses.</p>
        </div>

        <!-- Multi-Agent Chain-of-Thought Log -->
        <div class="bg-gray-900/90 border border-gray-800 rounded-xl p-4 flex flex-col">
          <h4 class="text-xs font-semibold uppercase text-cyan-400 mb-2 flex items-center justify-between">
            <span><i class="fa-solid fa-network-wired mr-1.5"></i>Multi-Agent Chain of Thought</span>
            <span class="text-[10px] text-gray-500 font-normal">Gemini Orchestrated</span>
          </h4>
          <div id="agentLogs" class="bg-black/50 p-3 rounded-lg flex-1 overflow-y-auto max-h-48 space-y-2 font-mono text-[11px]">
            <div class="text-blue-400">&gt; [Safety Agent]: Ingesting SPOT telemetry for WP05...</div>
            <div class="text-emerald-400">&gt; [Classifier]: Supervised Model predict major_liability=1 (P=0.985).</div>
            <div class="text-yellow-400">&gt; [Geospatial Agent]: Query Lake County GIS &bull; Karst Risk: SEVERE.</div>
            <div class="text-purple-400">&gt; [Actuarial Agent]: Quantifying net cost avoidance: $180,800.00.</div>
            <div class="text-rose-400">&gt; [AR HUD Agent]: Emitting 3D bounding packet to Apple Vision Pro.</div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    function updateVal(id, val) {
      document.getElementById(id).innerText = val;
    }

    function loadPreset() {
      const wp = document.getElementById("waypointSelect").value;
      if (wp === "WP05") {
        document.getElementById("visual_score").value = 18;
        document.getElementById("localization_pct").value = 65;
        document.getElementById("sound_db").value = 87;
        document.getElementById("karstReadout").innerText = "Severe Risk";
        document.getElementById("aadtReadout").innerText = "19,500";
      } else if (wp === "WP01") {
        document.getElementById("visual_score").value = 88;
        document.getElementById("localization_pct").value = 98;
        document.getElementById("sound_db").value = 54;
        document.getElementById("karstReadout").innerText = "Moderate";
        document.getElementById("aadtReadout").innerText = "48,500";
      }
      updateVal('visVal', document.getElementById("visual_score").value + '%');
      updateVal('locVal', document.getElementById("localization_pct").value + '%');
      updateVal('sndVal', document.getElementById("sound_db").value + ' dB');
    }

    async function runTriage() {
      const btn = document.getElementById("triageBtn");
      btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Autonomous Reasoning...`;
      btn.disabled = true;

      const wp = document.getElementById("waypointSelect").value;
      const vis = parseFloat(document.getElementById("visual_score").value);
      const loc = parseFloat(document.getElementById("localization_pct").value);
      const snd = parseFloat(document.getElementById("sound_db").value);

      const payload = {
        waypoint_id: wp,
        visual_score: vis,
        localization_pct: loc,
        sound_db: snd,
        has_anomaly: vis < 50 ? 1 : 0,
        failure_type: vis < 50 ? "STRUCTURAL_VOID" : "NONE",
        severity: vis < 35 ? "CRITICAL" : (vis < 60 ? "MEDIUM" : "LOW"),
        asset_criticality: vis < 35 ? "CRITICAL" : "LOW",
        status: vis < 40 ? "FAILED" : "SUCCESS"
      };

      try {
        const res = await fetch("/api/triage", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        console.error(err);
      } finally {
        btn.innerHTML = `<i class="fa-solid fa-play"></i> Trigger Autonomous Multi-Agent Triage`;
        btn.disabled = false;
      }
    }

    function renderResults(data) {
      const decision = data.triage_decision;
      const isMajor = decision.classification.major_liability;
      const prob = (decision.classification.confidence_score * 100).toFixed(1);
      const fin = decision.financial_actuarial_metrics;
      const ar = decision.smart_glasses_ar_hud;

      // Update Result Banner
      const banner = document.getElementById("resultBanner");
      const badge = document.getElementById("liabilityBadge");
      if (isMajor) {
        banner.className = "bg-gray-900/90 border-2 neon-border-red rounded-xl p-5 transition-all";
        badge.className = "px-3 py-1 rounded text-xs font-extrabold uppercase bg-red-600 text-white tracking-wide";
        badge.innerText = "MAJOR LIABILITY (CLASS 1)";
        document.getElementById("reticleBox").className = "hud-reticle w-24 h-16 rounded flex items-center justify-center border-red-500";
        document.getElementById("reticleIcon").className = "fa-solid fa-triangle-exclamation text-red-500 text-2xl";
      } else {
        banner.className = "bg-gray-900/90 border-2 neon-border-green rounded-xl p-5 transition-all";
        badge.className = "px-3 py-1 rounded text-xs font-extrabold uppercase bg-emerald-600 text-white tracking-wide";
        badge.innerText = "NON-MAJOR WEAR (CLASS 0)";
        document.getElementById("reticleBox").className = "w-24 h-16 rounded flex items-center justify-center border-2 border-emerald-500";
        document.getElementById("reticleIcon").className = "fa-solid fa-circle-check text-emerald-400 text-2xl";
      }

      document.getElementById("probBadge").innerText = `ML Confidence: ${prob}%`;
      document.getElementById("corridorHeading").innerText = decision.corridor_name;
      document.getElementById("recommendationText").innerText = decision.executive_recommendation;
      document.getElementById("costAvoidanceVal").innerText = `$${Number(fin.net_capital_cost_avoidance_usd).toLocaleString()}`;
      document.getElementById("roiVal").innerText = `Municipal ROI: ${(fin.municipal_roi_multiple || 1.0).toFixed(1)}x`;
      document.getElementById("tortVal").innerText = `$${Number(fin.expected_tort_liability_exposure_usd).toLocaleString()}`;

      // AR Updates
      if (ar && ar.hud_display) {
        document.getElementById("arBannerText").innerText = ar.hud_display.banner_title;
        document.getElementById("arBadgeText").innerText = ar.hud_display.status_badge;
      }

      // Chain of Thought Logs
      const logs = document.getElementById("agentLogs");
      logs.innerHTML = "";
      if (decision.chain_of_thought_trajectory) {
        decision.chain_of_thought_trajectory.forEach(step => {
          const div = document.createElement("div");
          if (step.action) {
            div.className = "text-blue-400";
            div.innerText = `> [${step.agent}]: ${step.action}`;
          } else {
            div.className = "text-emerald-400";
            div.innerText = `> [${step.agent}]: ${step.observation}`;
          }
          logs.appendChild(div);
        });
      }
    }
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
