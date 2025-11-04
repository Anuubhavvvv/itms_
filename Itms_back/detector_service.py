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
ocr_engine = SimpleOCRProcessor(use_gpu=False)
tracker = CentroidTracker(max_disappeared=50)

print("[INFO] All models initialized successfully.")


# ===========================================================
# Helper Functions
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


def expand_bbox(bbox, image_shape, scale=0.4):
    """Expand a bounding box by a percentage safely within image bounds."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * scale / 2), int(bh * scale / 2)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w - 1, x2 + pad_w)
    y2 = min(h - 1, y2 + pad_h)
    return [x1, y1, x2, y2]


# ===========================================================
# Core Function
# ===========================================================
def process_video(video_source=None):
    """
    Process a video or stream for ANPR.
    Ensures: one plate per vehicle, plate detected only inside vehicle crop.
    """

    # -------------------------------
    # Source setup
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

    # ===========================================================
    # Frame loop
    # ===========================================================
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
            x1, y1, x2, y2 = expand_bbox(v["bbox"], frame.shape, scale=0.4)
            conf = v.get("confidence", 0)
            v_crop = frame[y1:y2, x1:x2]
            vehicle_path = os.path.join(VEHICLE_DIR, f"{vid}_vehicle.jpg")

            # Save vehicle if new or higher confidence
            if objectID not in tracked or conf > tracked[objectID].get("vehicle_conf", 0):
                cv2.imwrite(vehicle_path, v_crop)
                tracked[objectID] = {
                    "vehicle_conf": conf,
                    "vehicle_path": vehicle_path,
                    "plate_path": "",
                    "plate_conf": 0,
                    "ocr_text": "",
                    "plate_assigned": False
                }

            # =======================================================
            # PLATE DETECTION (strictly inside vehicle)
            # =======================================================
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
            px1, py1, px2, py2 = expand_bbox(best_plate["bbox"], v_crop.shape, scale=0.4)
            plate_conf = best_plate["confidence"]

            # Skip duplicates or lower-confidence detections
            if tracked[objectID].get("plate_assigned") and plate_conf <= tracked[objectID].get("plate_conf", 0):
                continue

            # Physically ensure plate is inside vehicle crop (safety)
            vx1, vy1, vx2, vy2 = v["bbox"]
            if not (0 <= px1 < px2 <= (vx2 - vx1) and 0 <= py1 < py2 <= (vy2 - vy1)):
                continue

            # Crop from vehicle region only
            p_crop = v_crop[py1:py2, px1:px2]
            plate_path = os.path.join(PLATE_DIR, f"{vid}_plate.jpg")
            cv2.imwrite(plate_path, p_crop)

            tracked[objectID].update({
                "plate_assigned": True,
                "plate_conf": plate_conf,
                "plate_path": plate_path
            })

            # =======================================================
            # OCR (Hybrid: Paddle primary, Easy fallback)
            # =======================================================
            try:
                prepped = preprocess_plate(p_crop)
                ocr_result = ocr_engine.extract_text_from_image(prepped)
                ocr_text = " ".join(ocr_result.get("detected_text", [])).strip()
            except Exception as e:
                ocr_text = ""
                print(f"[WARN] OCR failed → {e}")

            # Only update if OCR text is new or improved
            if ocr_text and ocr_text != tracked[objectID].get("ocr_text", ""):
                tracked[objectID]["ocr_text"] = ocr_text

                detection = {
                    "id": vid,
                    "timestamp": datetime.now().isoformat(),
                    "vehicle_conf": conf,
                    "plate_conf": plate_conf,
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
# Entry Point
# ===========================================================
if __name__ == "__main__":
    try:
        process_video()
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[FATAL] Unhandled error → {e}")
