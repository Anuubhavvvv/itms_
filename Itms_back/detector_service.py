# detector_service.py
# ---------------------------------------------------------------------
# Local ANPR detection service that saves images and POSTs data to FastAPI
# ---------------------------------------------------------------------

import cv2
import os
import requests
import json
from datetime import datetime
from processors.vehicle_detect import VehicleDetector
from processors.anpr_detect import ANPRDetector
from processors.ocr_method import SimpleOCRProcessor
from config import (
    VEHICLE_CONFIDENCE,
    PLATE_CONFIDENCE,
    VEHICLE_MODEL_PATH,
    PLATE_MODEL_PATH,
    OCR_LANGUAGE,
    VEHICLE_DIR,
    PLATE_DIR,
    API_DETECTION_ENDPOINT,
)

# Ensure output folders exist
os.makedirs(VEHICLE_DIR, exist_ok=True)
os.makedirs(PLATE_DIR, exist_ok=True)

# Initialize detectors
vehicle_detector = VehicleDetector(VEHICLE_MODEL_PATH)
plate_detector = ANPRDetector(PLATE_MODEL_PATH)
ocr_engine = SimpleOCRProcessor(OCR_LANGUAGE)

def generate_id():
    now = datetime.now()
    date_tag = now.strftime("%d%m%Y")
    time_tag = now.strftime("%H%M%S%f")[-6:]
    return f"{date_tag}_{time_tag}"

def process_video(video_source="test_video.avi"):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_source}")
        return

    print("[INFO] Starting detection stream...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        vehicles = vehicle_detector.detect_vehicles(rgb, VEHICLE_CONFIDENCE)
        plates = plate_detector.detect_number_plates(rgb, PLATE_CONFIDENCE)

        for v in vehicles:
            vid = generate_id()
            vconf = v["confidence"]
            vbbox = [int(x) for x in v["bbox"]]
            x1, y1, x2, y2 = vbbox
            v_crop = frame[y1:y2, x1:x2]

            vehicle_path = os.path.join(VEHICLE_DIR, f"{vid}_vehicle.jpg")
            cv2.imwrite(vehicle_path, v_crop)

            # Find nearest plate for this vehicle
            p_best, best_d = None, float("inf")
            for p in plates:
                px1, py1, px2, py2 = p["bbox"]
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                vcx, vcy = (x1 + x2) / 2, (y1 + y2) / 2
                d = ((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5
                if d < best_d:
                    best_d, p_best = d, p

            pconf, ocr_text, plate_path = 0.0, "", ""
            if p_best and best_d < 250:
                px1, py1, px2, py2 = [int(x) for x in p_best["bbox"]]
                p_crop = frame[py1:py2, px1:px2]
                plate_path = os.path.join(PLATE_DIR, f"{vid}_plate.jpg")
                cv2.imwrite(plate_path, p_crop)

                # Run OCR immediately
                ocr_result = ocr_engine.extract_text_from_image(p_crop)
                if "error" not in ocr_result:
                    texts = ocr_result.get("detected_text", [])
                    confs = ocr_result.get("confidence_scores", [])
                    ocr_text = " ".join(texts)
                    pconf = max(confs) if confs else 0.0

            # Build payload
            detection = {
                "id": vid,
                "timestamp": datetime.now().isoformat(),
                "vehicle_conf": vconf,
                "plate_conf": pconf,
                "ocr_text": ocr_text,
                "vehicle_path": vehicle_path,
                "plate_path": plate_path,
            }

            # POST to API
            try:
                requests.post(API_DETECTION_ENDPOINT, json=detection, timeout=2)
                print(f"[POSTED] {vid} OCR='{ocr_text}'")
            except Exception as e:
                print(f"[WARN] Failed to post detection: {e}")

        if frame_count % 30 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    cap.release()
    print("[INFO] Finished video processing.")
