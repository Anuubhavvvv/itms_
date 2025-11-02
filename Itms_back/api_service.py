# api_service.py
# ---------------------------------------------------------------------
# FastAPI backend for the local ANPR system
# ---------------------------------------------------------------------

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import Detection, SessionLocal, init_db
from config import VEHICLE_DIR, PLATE_DIR, RESULTS_DIR, API_HOST, API_PORT
import os

app = FastAPI(title="ANPR API Service")

# Allow GUI or local apps to access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ----------------------- ROUTES ---------------------------

@app.post("/detections")
async def add_detection(detection: dict):
    """Receive detection data from detector_service"""
    db: Session = SessionLocal()
    try:
        data = Detection(
            id=detection.get("id"),
            timestamp=detection.get("timestamp"),
            vehicle_conf=detection.get("vehicle_conf"),
            plate_conf=detection.get("plate_conf"),
            ocr_text=detection.get("ocr_text"),
            vehicle_path=detection.get("vehicle_path"),
            plate_path=detection.get("plate_path"),
        )
        db.add(data)
        db.commit()
        db.refresh(data)
        return {"status": "success", "id": data.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/detections")
async def get_all_detections():
    """Return all detection records"""
    db: Session = SessionLocal()
    try:
        data = db.query(Detection).all()
        return [d.__dict__ for d in data]
    finally:
        db.close()


@app.get("/detections/{detection_id}")
async def get_detection(detection_id: str):
    """Fetch single detection record"""
    db: Session = SessionLocal()
    try:
        record = db.query(Detection).filter(Detection.id == detection_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Detection not found")
        return record.__dict__
    finally:
        db.close()


@app.get("/images/{folder}/{filename}")
async def serve_image(folder: str, filename: str):
    """Serve saved images (vehicle or plate)"""
    if folder not in ["vehicles", "plates"]:
        raise HTTPException(status_code=400, detail="Invalid folder")
    file_path = os.path.join(RESULTS_DIR, folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# ----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
