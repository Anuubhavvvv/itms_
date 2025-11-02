# config.py
# ---------------------------------------------------------------------
# Central configuration file for the local ANPR backend system
# ---------------------------------------------------------------------

# PostgreSQL connection string (fill this in)
# Example: "postgresql+psycopg2://postgres:password@localhost:5432/anpr_db"
DB_URL = ""

# Detection confidence thresholds
VEHICLE_CONFIDENCE = 0.5
PLATE_CONFIDENCE = 0.5

# Directory paths
RESULTS_DIR = "results"
VEHICLE_DIR = f"{RESULTS_DIR}/vehicles"
PLATE_DIR = f"{RESULTS_DIR}/plates"

# FastAPI server host and port
API_HOST = "0.0.0.0"
API_PORT = 8000

# API endpoint for detector to send data
API_DETECTION_ENDPOINT = f"http://{API_HOST}:{API_PORT}/detections"

# Model paths (fill in with actual weights if needed)
VEHICLE_MODEL_PATH = "models/vehicle_yolov8.pt"
PLATE_MODEL_PATH = "models/anpr_best.pt"

# OCR language
OCR_LANGUAGE = "en"
