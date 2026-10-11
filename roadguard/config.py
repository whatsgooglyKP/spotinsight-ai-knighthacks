"""
RoadGuard Agent: System Configuration & Lake County GIS Metadata.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(
    r"C:\Users\pinar\Downloads\SPOT_KnightHacks_Challenge_Kit_Part_One (1)\SPOT_KnightHacks_Challenge_Kit Part One\02_DATA"
)
DEFAULT_ASSETS_DIR = Path(
    r"C:\Users\pinar\Downloads\SPOT_KnightHacks_Challenge_Kit_Part_One (1)\SPOT_KnightHacks_Challenge_Kit Part One\04_ASSETS"
)
MODELS_DIR = BASE_DIR / "saved_models"
OUTPUT_DIR = BASE_DIR / "reports"
DEFAULT_MODEL_PATH = MODELS_DIR / "roadguard_classifier.pkl"
SAVED_MODEL_PATH = DEFAULT_MODEL_PATH

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lake County, FL Roadway & Infrastructure Waypoint Mapping
LAKE_COUNTY_WAYPOINTS = {
    "WP01": {
        "corridor_name": "US-27 South Lake Logistics Corridor (Four Corners)",
        "zone_type": "ACCESS",
        "road_classification": "Principal Arterial / Freight Corridor",
        "annual_daily_traffic": 48500,
        "flood_zone": "X (Low-Moderate)",
        "karst_subsidence_risk": "Moderate",
        "primary_risk": "Heavy truck rutting & pavement shear",
        "speed_limit_mph": 65,
    },
    "WP02": {
        "corridor_name": "SR-19 Lake Harris Causeway & Bridge",
        "zone_type": "ELECTRICAL",
        "road_classification": "State Highway / Lakefront Causeway",
        "annual_daily_traffic": 24000,
        "flood_zone": "AE (High Storm Surge / 100-yr Flood)",
        "karst_subsidence_risk": "Low (Alluvial Silt)",
        "primary_risk": "Storm surge scour, thermal joint expansion, electrical sensor short",
        "speed_limit_mph": 55,
    },
    "WP03": {
        "corridor_name": "Eustis Ridge Stormwater Trunk & Culvert",
        "zone_type": "PROCESS",
        "road_classification": "Major Collector / Drainage Interceptor",
        "annual_daily_traffic": 16800,
        "flood_zone": "A (Frequent Inundation)",
        "karst_subsidence_risk": "High (Limestone Cavity)",
        "primary_risk": "Subgrade void piping & culvert joint separation",
        "speed_limit_mph": 45,
    },
    "WP04": {
        "corridor_name": "Tavares Dead River Bridge Approach",
        "zone_type": "MECHANICAL",
        "road_classification": "Urban Arterial Bridge",
        "annual_daily_traffic": 31200,
        "flood_zone": "AE (High Flood Hazard)",
        "karst_subsidence_risk": "Moderate",
        "primary_risk": "Bridge deck vibration scour & bearing displacement",
        "speed_limit_mph": 45,
    },
    "WP05": {
        "corridor_name": "Mount Dora Karst Pipe Gallery / Subgrade Void Zone",
        "zone_type": "PROCESS",
        "road_classification": "High-Failure Historical Subgrade Corridor",
        "annual_daily_traffic": 19500,
        "flood_zone": "X (Special Karst Basin)",
        "karst_subsidence_risk": "Severe (Active Karst Sinkhole Vulnerability)",
        "primary_risk": "Catastrophic roadway sinkhole subsidence & subgrade collapse",
        "speed_limit_mph": 35,
    },
    "WP06": {
        "corridor_name": "Groveland Heavy Haul Bypass (SR-50 West)",
        "zone_type": "MECHANICAL",
        "road_classification": "Heavy Haul Mining & Freight Route",
        "annual_daily_traffic": 38000,
        "flood_zone": "X",
        "karst_subsidence_risk": "Moderate",
        "primary_risk": "Deep sub-base fatigue cracking & axle overload shear",
        "speed_limit_mph": 60,
    },
    "WP07": {
        "corridor_name": "Clermont Lake Minneola Scenic Byway",
        "zone_type": "ACCESS",
        "road_classification": "Urban Scenic Arterial / Multimodal Corridor",
        "annual_daily_traffic": 14200,
        "flood_zone": "X",
        "karst_subsidence_risk": "Low",
        "primary_risk": "Pedestrian/cyclist collision hazard & visual obstruction",
        "speed_limit_mph": 30,
    },
    "WP08": {
        "corridor_name": "Leesburg Rail-Highway Crossing (US-441)",
        "zone_type": "MECHANICAL",
        "road_classification": "Principal Arterial at Rail Intermodal Hub",
        "annual_daily_traffic": 36000,
        "flood_zone": "X",
        "karst_subsidence_risk": "Low",
        "primary_risk": "Flangeway gap hazard & acoustic rail dynamic shock",
        "speed_limit_mph": 50,
    },
    "WP09": {
        "corridor_name": "Fruitland Park North Arterial Corridor",
        "zone_type": "MECHANICAL",
        "road_classification": "Suburban Divided Arterial",
        "annual_daily_traffic": 22000,
        "flood_zone": "X",
        "karst_subsidence_risk": "Low",
        "primary_risk": "Surface friction loss & stormwater retention pump failure",
        "speed_limit_mph": 55,
    },
    "WP10": {
        "corridor_name": "Sorrento/Wekiva Parkway Environmental Link",
        "zone_type": "PROCESS",
        "road_classification": "Ecological Tollway Connector",
        "annual_daily_traffic": 18500,
        "flood_zone": "A (Environmentally Sensitive Wetland)",
        "karst_subsidence_risk": "Moderate",
        "primary_risk": "Culvert flap-valve blockage & eco-system stormwater breach",
        "speed_limit_mph": 65,
    },
    "WP11": {
        "corridor_name": "Lake County Central Fleet Depot & Charging Hub",
        "zone_type": "ELECTRICAL",
        "road_classification": "Municipal Fleet Staging / Autonomous Dock",
        "annual_daily_traffic": 2500,
        "flood_zone": "X",
        "karst_subsidence_risk": "Low",
        "primary_risk": "Spot battery thermal strain & mission abort risk",
        "speed_limit_mph": 15,
    },
    "WP12": {
        "corridor_name": "Minneola Turnpike Interchange Flyover",
        "zone_type": "PROCESS",
        "road_classification": "Elevated Expressway Ramp",
        "annual_daily_traffic": 42000,
        "flood_zone": "X",
        "karst_subsidence_risk": "Low",
        "primary_risk": "Flyover pier thermal expansion strain & drainage blockage",
        "speed_limit_mph": 55,
    },
    "WP13": {
        "corridor_name": "Lady Lake / The Villages Commercial Corridor",
        "zone_type": "ACCESS",
        "road_classification": "High-Density Commercial Arterial",
        "annual_daily_traffic": 29000,
        "flood_zone": "X",
        "karst_subsidence_risk": "Moderate",
        "primary_risk": "Utility trench differential settlement & golf cart joint hazards",
        "speed_limit_mph": 40,
    },
    "WP14": {
        "corridor_name": "Astor St. Johns River Floodway Embankment",
        "zone_type": "MECHANICAL",
        "road_classification": "Rural Riverine Causeway",
        "annual_daily_traffic": 8900,
        "flood_zone": "AE (High Velocity Flood Hazard)",
        "karst_subsidence_risk": "High (Muck & Saturation)",
        "primary_risk": "Embankment slope slumping & chronic hydrostatic road breach",
        "speed_limit_mph": 45,
    },
    "WP15": {
        "corridor_name": "Tavares County Operations Field Staging Deck",
        "zone_type": "ACCESS",
        "road_classification": "Municipal Operations Command Yard",
        "annual_daily_traffic": 3000,
        "flood_zone": "X",
        "karst_subsidence_risk": "Low",
        "primary_risk": "Edge mission telemetry ingest & drone sensor calibration",
        "speed_limit_mph": 15,
    },
}

# Financial & Actuarial Parameters (Lake County Civil & Municipal Risk)
FINANCIAL_PARAMS = {
    # Florida Sovereign Immunity / Municipal Tort Liability Claim Risk (FL Stat § 768.28)
    "avg_catastrophic_tort_claim": 350000.0,
    "severe_accident_liability_max": 1500000.0,
    # Emergency Road Repair / Emergency Closure vs Proactive Repair
    "emergency_road_closure_cost": 185000.0,
    "proactive_robotic_repair_cost": 4200.0,
    "net_repair_cost_avoidance": 180800.0,
    # Pavement Life Extension & Carbon Economics
    "carbon_cost_per_ton": 55.0,  # EPA social cost of carbon
    "emissions_avoidance_kg_co2_per_vmt": 0.042,  # Smoother pavement vs rough damaged road
    "insurance_premium_credit_rate": 0.085,  # 8.5% municipal property/casualty pool credit for robotic monitoring
}
