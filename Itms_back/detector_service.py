import os
import cv2
import requests
from datetime import datetime
from processors.vehicle_detect import VehicleDetector
from processors.anpr_detect import ANPRDetector
from processors.ocr_method import SimpleOCRProcessor
from processors.centroid_tracker import CentroidTracker
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

# ===========================================================
# Initialization
# ===========================================================
os.makedirs(VEHICLE_DIR, exist_ok=True)
os.makedirs(PLATE_DIR, exist_ok=True)
os.makedirs("debug", exist_ok=True)

vehicle_detector = VehicleDetector(VEHICLE_MODEL_PATH)
plate_detector = ANPRDetector(PLATE_MODEL_PATH)
ocr_engine = SimpleOCRProcessor(OCR_LANGUAGE, use_gpu=False)
tracker = CentroidTracker(maxDisappeared=50)

print("[INFO] All models initialized successfully.")


# ===========================================================
# Helper
# ===========================================================
def preprocess_plate(img):
    """Mild denoise and upscale before OCR."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    if h < 150:
        scale = 200 / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ===========================================================
# Core Function
# ===========================================================
def process_video(video_source=None):
    """
    Process a video or stream for ANPR.
    Supports:
      - Local path (e.g. test_video.avi)
      - Webcam index (0)
      - Network streams (rtsp/http)
    """

    # -------------------------------
    # Source resolution
    # -------------------------------
    if video_source is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        video_source = os.path.join(base_dir, "test_video.avi")

    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    if isinstance(video_source, str):
        if os.path.exists(video_source):
            print(f"[INFO] Using local video file: {video_source}")
        elif video_source.startswith(("rtsp://", "http://", "https://")):
            print(f"[INFO] Using network stream: {video_source}")
        else:
            print(f"[ERROR] Invalid video path: {video_source}")
            return
    elif isinstance(video_source, int):
        print(f"[INFO] Using webcam index: {video_source}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[FATAL] Cannot open video source: {video_source}")
        return

    print("[INFO] Detection stream started...")
    frame_count = 0
    tracked = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream or read error.")
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # VEHICLE DETECTION
        try:
            vehicles = vehicle_detector.detect_vehicles(rgb, VEHICLE_CONFIDENCE)
        except Exception as e:
            print(f"[ERROR] Vehicle detection failed → {e}")
            continue

        rects = [v["bbox"] for v in vehicles]
        objects = tracker.update(rects)

        for (objectID, centroid), v in zip(objects.items(), vehicles):
            vid = f"{datetime.now().strftime('%d%m%Y')}_{objectID:03d}"
            x1, y1, x2, y2 = [int(x) for x in v["bbox"]]
            conf = v.get("confidence", 0)
            h, w, _ = frame.shape

            # expand crop slightly
            pad_w, pad_h = int(0.3 * (x2 - x1)), int(0.3 * (y2 - y1))
            x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
            v_crop = frame[y1:y2, x1:x2]
            vehicle_path = os.path.join(VEHICLE_DIR, f"{vid}_vehicle.jpg")

            if objectID not in tracked or conf > tracked[objectID]["vehicle_conf"]:
                cv2.imwrite(vehicle_path, v_crop)
                tracked[objectID] = {
                    "vehicle_conf": conf,
                    "vehicle_path": vehicle_path,
                    "plate_path": "",
                    "ocr_text": "",
                }

            # PLATE DETECTION
            try:
                roi_rgb = cv2.cvtColor(v_crop, cv2.COLOR_BGR2RGB)
                local_plates = plate_detector.detect_number_plates(
                    roi_rgb, max(0.25, PLATE_CONFIDENCE - 0.1)
                )
            except Exception as e:
                print(f"[WARN] Plate detection failed → {e}")
                continue

            if not local_plates:
                continue

            best_plate = max(local_plates, key=lambda p: p["confidence"])
            px1, py1, px2, py2 = [int(x) for x in best_plate["bbox"]]
            p_crop = v_crop[py1:py2, px1:px2]
            plate_path = os.path.join(PLATE_DIR, f"{vid}_plate.jpg")
            cv2.imwrite(plate_path, p_crop)

            # OCR
            try:
                prepped = preprocess_plate(p_crop)
                ocr_result = ocr_engine.extract_text_from_image(prepped, vehicle_id=vid)
                ocr_text = " ".join(ocr_result.get("detected_text", [])).strip()
            except Exception as e:
                ocr_text = ""
                print(f"[WARN] OCR failed → {e}")

            if not tracked[objectID]["ocr_text"] or ocr_text != tracked[objectID]["ocr_text"]:
                tracked[objectID]["plate_path"] = plate_path
                tracked[objectID]["ocr_text"] = ocr_text

                detection = {
                    "id": vid,
                    "timestamp": datetime.now().isoformat(),
                    "vehicle_conf": conf,
                    "plate_conf": best_plate["confidence"],
                    "ocr_text": ocr_text,
                    "vehicle_path": vehicle_path,
                    "plate_path": plate_path,
                }

                try:
                    r = requests.post(API_DETECTION_ENDPOINT, json=detection, timeout=3)
                    print(f"[POST] Vehicle {objectID} | OCR='{ocr_text}' | HTTP {r.status_code}")
                except Exception as e:
                    print(f"[ERROR] POST failed → {e}")

        if frame_count % 30 == 0:
            print(f"[INFO] Processed {frame_count} frames | Active IDs: {list(objects.keys())}")

    cap.release()
    print("[INFO] Video processing complete.")


# ===========================================================
# Entry point
# ===========================================================
if __name__ == "__main__":
    try:
        process_video()
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[FATAL] Unhandled error → {e}")
