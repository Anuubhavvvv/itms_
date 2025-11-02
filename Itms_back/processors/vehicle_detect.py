import cv2
import numpy as np
from ultralytics import YOLO
import torch


class VehicleDetector:
    def __init__(self, model_path="C:\\Itms_back\\models\\vehicle_yolov8.pt"):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detect_vehicles(self, image, confidence_threshold=0.5):
        if hasattr(image, "convert"):
            image = np.array(image)
        results = self.model.predict(source=image, conf=confidence_threshold, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": self.model.names[cls]
                })
        return detections
