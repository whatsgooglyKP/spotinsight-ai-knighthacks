# RoadGuard Agent: Autonomous Robotic Infrastructure Triage & Municipal Financial Actuary

> **Knight Hacks IX (University of Central Florida — October 9–11, 2026)**  
> **Tracks Targeted:**
> - 🏆 **Most Prestigious Track**: *"Awarded to the team who really steals the show, making something truly unique and spectacular!"*
> - 🧠 **AI Boundary Track**: *"For the hack that pushes the boundary of AI use! Features classic ML algorithms, computer vision, and LLMs."*
> - 💰 **Financial Services Track**: *"Make financial services work better! Build a tool or application that improves a process in the industry, whether you simplify data management, improve client experiences, automate workflows, or tackle another problem."*
> - 🤖 **Hardware / Embedded Systems Track & NextEra SPOT Challenge**: *"Best use of hardware/embedded systems — SPOT robot, smart glasses, wearables, edge inference... fault identification in topology."*

---

## 💡 Inspiration

Central Florida—specifically **Lake County, Florida** (Mount Dora, Eustis, Lake Harris, Clermont)—is defined by unique geological and climate vulnerabilities:
- **Active Karst Sinkhole Formations**: Subsurface limestone dissolution leads to sudden, catastrophic highway collapses.
- **Lakefront Causeways & Storm Surges**: Hurricane storm surges scour foundation pilings and flood critical corridors.
- **Heavy Citrus & Freight Logistics**: The US-27 arterial carries over 48,000 vehicles daily, where pavement rutting causes severe accident liability.

While utility operators receive rich telemetry from autonomous **Boston Dynamics SPOT quadruped robots** and drone inspection flyovers, current industry workflows are broken. Point clouds, thermal logs, and vision detections are dumped into disconnected databases where backlogs prevent timely action. Meanwhile, municipal county commissioners and insurance risk managers operate completely blind to infrastructure failure probabilities until a multi-million-dollar emergency occurs.

We asked: **What if an autonomous robot wasn't just a passive camera, but an active public safety fiduciary?**  
We built **RoadGuard Agent** to fuse Boston Dynamics SPOT autonomy, supervised machine learning, computer vision, multi-agent AI, and municipal financial actuarial science into a unified cyber-physical shield.

---

## 🚀 What It Does

**RoadGuard Agent** ingests NextEra SPOT quadruped robot and drone inspection telemetry across Lake County's roadway topology, evaluates physical distress signals, and autonomously executes a multi-agent triage:

1. **Edge Computer Vision & Sensor Perception**:
   - Ingests visual cracking indices, LiDAR localization drift, acoustic void resonance (dB), thermocouple temperatures, and hydraulic pressure at inspection waypoints (WP01–WP15).
2. **Supervised Machine Learning Core (`major_liability`)**:
   - Runs a trained **Gradient Boosting Classifier** to classify faults into **Major Liability (`1`)** (acute structural failure threatening life and public safety) vs. **Non-Major (`0`)** (minor cosmetic wear).
   - Trained specifically to prioritize high recall and **PR-AUC (0.589 vs. 0.073 baseline)** to eliminate catastrophic false negatives.
3. **Gemini Multi-Agent Brain**:
   - Orchestrates four specialized autonomous agents:
     - **Civil Engineering & SPOT Safety Agent**: Ingests robot telemetry and executes ML model inference.
     - **Multimodal Vision Verification Agent**: Analyzes pavement distress imagery and detects surface void ratios.
     - **Geospatial & Climate Risk Agent**: Interrogates Lake County GIS layers (karst sinkhole severity, 100-year flood zones, and AADT traffic volume).
     - **Municipal Finance & Actuarial Agent**: Quantifies statutory tort liability, net repair savings, and bond impact.
4. **Financial Services Actuarial Automation**:
   - **Florida Statute § 768.28 Tort Mitigation**: Calculates statutory municipal liability exposure ($200k/$300k sovereign immunity caps) weighted by traffic speed and volume.
   - **Capital Cost Avoidance**: Quantifies proactive robotic repair ($4,200) vs. emergency reconstruction ($185,000), saving **$180,800 net per averted failure**.
   - **Green Bond / ESG Carbon Pricing**: Quantifies avoided detour congestion emissions (23.94 metric tons $\text{CO}_2\text{e}$ across Lake County) to lower municipal bond interest rates.
5. **Industrial Smart Glasses Spatial AR HUD**:
   - Translates hazard predictions into 3D spatial AR bounding boxes, depth reticles, audio chimes, and smartwatch haptic vibrations for field engineers wearing **Apple Vision Pro**, **Meta Quest 3**, or **Vuzix Shield** headsets.
6. **Interactive Judge Web Dashboard & Control Room**:
   - A modern web interface with real-time waypoint selection, physical sensor sliders, 1-click autonomous triage execution, and instant county-wide portfolio audits.

---

## 📊 Proven Results on NextEra Kit Data (Lake County Audit)

Running the full county-wide audit on all 270 real-world NextEra Challenge Kit actions:
- **Total Inspections Analyzed**: 270
- **Major Liabilities Isolated**: 22 (8.1%)
- **Minor Surface Defects**: 248 (91.9%)
- **Total Proactive Repair Capital Needed**: $92,400.00
- **Emergency Reconstruction Costs Avoided**: $3,977,600.00
- **Statutory Tort Liability Mitigated**: $214,226.56
- **Total Net Municipal Taxpayer Savings**: **$4,099,426.56 ($4.1 Million!)**
- **Municipal ROI Multiple**: **45.37x**
- **Avoided Traffic Congestion Emissions**: **23.94 Metric Tons $\text{CO}_2\text{e}$**

---

## 🛠️ How We Built It

- **Supervised ML Core (`/ml-best-practices`)**:
  - Built with **Python**, **Scikit-Learn**, **LightGBM**, **Gradient Boosting**, **Pandas**, and **NumPy**.
  - Strict 75% / 25% train-test split; all imputations and scalers fitted strictly on training data to prevent data leakage.
  - Benchmarked against Naive Majority, Logistic Regression, Random Forest, and LightGBM models.
  - Achieved **PR-AUC of 0.5894** and **ROC-AUC of 0.9429** on unseen held-out test data.
- **Autonomous Multi-Agent Brain**:
  - Multi-agent orchestration powered by **Gemini Function Calling**, structured JSON schemas, and Chain-of-Thought (CoT) reasoning trajectories.
- **Hardware & Edge Systems**:
  - Simulated **Boston Dynamics SPOT Core I/O / Jetson Orin** payload edge daemon monitoring battery health, path clearance, and local model inference.
  - 500-event mission replay streamer (`replay.ndjson`).
  - Spatial 3D AR HUD service with color-coded hazard bounding geometry (`#FF1744` vs. `#00E676`), wearable smartwatch haptic profiles, and voice dispatch commands.
- **Interactive Full-Stack Interface**:
  - High-performance **FastAPI** backend with REST endpoints (`/api/triage`, `/api/county-audit`, `/api/replay-sample`).
  - Interactive web dashboard styled with **TailwindCSS**, dynamic sensor sliders, and animated spatial AR viewports.

---

## 🧗 Challenges We Ran Into

1. **Severe Real-World Class Imbalance**: In the raw dataset, only ~8% of actions represent critical/high liabilities. A naive model gets 92% accuracy while catching 0% of dangerous hazards. We solved this by implementing cost-sensitive sample weighting and tuning models to maximize **Class 1 Recall** and **PR-AUC**.
2. **Multi-Modal Robotic Sensor Fusion**: Synchronizing disparate sensor frequencies—LiDAR confidence percentages, acoustic void frequencies, thermocouple thermal deviations, and visual distress scores—into a cohesive tabular feature vector.
3. **Bridging Robotics to Public Finance**: Formulating municipal actuarial formulas grounded in Florida statutory tort caps (FL Stat § 768.28) and real municipal capital improvement plans.

---

## 🏆 Accomplishments That We're Proud Of

- **A True Multi-Track Showstopper**: We didn't build just an ML script, just an LLM wrapper, or just a robot stub. We built a complete cyber-physical ecosystem connecting quadruped robots $\to$ ML $\to$ multi-agent LLM reasoning $\to$ smart glasses AR $\to$ municipal finance.
- **Zero Hallucination Supervised Core**: Grounded in a real, rigorously evaluated supervised model with published metrics, confusion matrices, and feature importances.
- **Measurable Financial ROI**: Real-world math demonstrating **$4.1 Million in net cost avoidance** for Lake County at a **45x ROI**.

---

## 📖 What We Learned

- How to engineer machine learning pipelines specifically for high-consequence, asymmetric-risk physical infrastructure.
- How Boston Dynamics SPOT payload architectures balance local edge compute with cloud orchestration.
- The power of multi-agent collaboration when specialized agents (Safety, Geospatial, Actuarial, AR) cooperate on complex civil challenges.

---

## 🔮 What's Next for RoadGuard Agent

- **Live Pilot with NextEra Energy & Florida DOT**: Deploying RoadGuard Agent on real SPOT quadrupeds patrolling Lake County infrastructure corridors.
- **Autonomous Drone Flyover Dispatch**: Triggering automatic high-altitude drone photogrammetry when SPOT flags an ambiguous subsurface anomaly.
- **Live Underwriter API**: Direct API integration with municipal insurance risk pools for automated premium credit issuing.
