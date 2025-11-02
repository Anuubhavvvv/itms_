import cv2
import numpy as np
from ultralytics import YOLO
import torch


class ANPRDetector:
    def __init__(self, model_path="C:\\Itms_back\\models\\anpr_best.pt"):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detect_number_plates(self, image, confidence_threshold=0.5):
        """
        Detect number plates using YOLOv8 .predict()
        Returns list of dicts with bbox, confidence, and class_name.
        """
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
                    "class_name": self.model.names[cls] if hasattr(self.model, "names") else "plate"
                })
        return detections

    def crop_number_plates(self, image, detections):
        """Crop plate ROIs from image based on YOLO detections."""
        if hasattr(image, "convert"):
            image = np.array(image)

        crops = []
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            crop = image[y1:y2, x1:x2]
            crops.append({
                "image": crop,
                "bbox": d["bbox"],
                "confidence": d["confidence"],
                "class_name": d["class_name"]
            })
        return crops
