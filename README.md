# SPOTInsight AI — KnightHacks 2026 Winning Submission

**Tracks targeted:**
- NextEra Energy: SPOT Management Dashboard (Internship interview)
- MLH Best Use of Gemini API (Swag Kits)
- OneEthos: Innovation for Good (CEO interview + prizes)

## Problem
Energy plant operators receive rich Boston Dynamics SPOT robot inspection data (missions, actions, sensor readings, anomalies, locations, assets) but struggle to quickly understand what matters, where issues are, why they matter, and what to do next. Non-technical operators need confidence without becoming data experts.

Simultaneously, Central Florida communities (Lake County / Eustis area) face environmental challenges around lake water quality and the risk that energy infrastructure failures can affect local ecosystems. Reliable, proactive inspection reduces that risk.

## Solution
SPOTInsight AI is a Streamlit dashboard that:
- Visualizes waypoints and anomalies on a facility map
- Provides a prioritized anomaly triage queue
- Compares BASELINE (V1) vs IMPROVED (V2) performance
- Explores sensor readings over time
- Uses carefully engineered Gemini prompts to generate plain-language explanations, operator briefings, and community impact narratives
- Quantifies measurable environmental co-benefits (success rate improvement, high-severity anomaly reduction)

## Key Insights from Data
- IMPROVED configuration shows higher mission completion (~100% vs ~94%) and action success (~91% vs ~77%)
- High/critical anomalies are actionable and concentrated on specific assets/locations
- Replay stream and sample images support live-style and multimodal extensions

## Run Locally
```bash
cd spotinsight
pip install -r requirements.txt
streamlit run app.py
```

## Gemini Integration
All major insights have ready-to-use prompts in the Gemini Assistant and Community Impact tabs. Connect a Gemini API key (google-generativeai) for live responses. Prompts are designed for structured, operator-friendly, and community-accessible output.

## Submission Notes
- Data is the official synthetic KnightHacks SPOT challenge kit
- All operational claims are supported by the provided CSVs and validation_results.json
- Designed for non-technical operators while delivering deep analytical value
- Explicit local Florida environmental framing for OneEthos track

Built as a hackathon judge expert submission to win all three tracks.
