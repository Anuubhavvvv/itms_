import cv2
import json
from paddleocr import PaddleOCR


class SimpleOCRProcessor:
    def __init__(self, lang="en"):
        # do not use show_log; keep simple compatible init
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        except Exception:
            # fallback to basic init if some builds differ
            self.ocr = PaddleOCR(lang=lang)

    def preprocess_plate(self, plate_image):
        if plate_image is None:
            return None
        img = plate_image
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.equalizeHist(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def extract_text_from_image(self, image):
        """
        image: BGR numpy array
        returns: dict with keys 'detected_text' and 'confidence_scores' or {'error': str}
        """
        img = self.preprocess_plate(image)
        if img is None:
            return {"detected_text": [], "confidence_scores": []}
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ocr_result = self.ocr.ocr(img_rgb)
        except Exception as e:
            return {"error": str(e)}
        detected_text = []
        confidences = []
        if not ocr_result:
            return {"detected_text": [], "confidence_scores": []}
        try:
            for line in ocr_result[0]:
                text = line[1][0]
                conf = float(line[1][1])
                detected_text.append(text)
                confidences.append(conf)
        except Exception:
            return {"detected_text": [], "confidence_scores": []}
        return {"detected_text": detected_text, "confidence_scores": confidences}

    def ocr_and_save_json(self, plate_path, out_json_path, id_label=None):
        img = cv2.imread(plate_path)
        res = self.extract_text_from_image(img)
        payload = {"id": id_label, "image_path": plate_path, **res}
        tmp = out_json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        # atomic replace
        import os
        os.replace(tmp, out_json_path)
        return payload
