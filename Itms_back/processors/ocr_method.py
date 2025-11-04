import cv2
import numpy as np
import re
import os
from paddleocr import PaddleOCR
import easyocr

class SimpleOCRProcessor:
    def __init__(self, use_gpu=False):
        """
        Hybrid OCR engine: PaddleOCR (primary) + EasyOCR (fallback)
        - Uses Paddle for accuracy
        - Falls back to EasyOCR if Paddle fails or has low confidence
        """
        self.paddle = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
        self.easy = easyocr.Reader(['en'], gpu=use_gpu)
        os.makedirs("debug", exist_ok=True)
        print(f"[INFO] Hybrid OCR initialized (GPU={use_gpu})")

    # -----------------------------------------------------------
    # Preprocessing before OCR
    # -----------------------------------------------------------
    def _preprocess(self, img):
        if img is None or img.size == 0:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.equalizeHist(gray)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)

        thresh = cv2.adaptiveThreshold(
            sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9
        )
        cv2.imwrite("debug/ocr_preprocessed.jpg", thresh)
        return thresh

    # -----------------------------------------------------------
    # Main OCR logic
    # -----------------------------------------------------------
    def extract_text_from_image(self, img):
        processed = self._preprocess(img)
        if processed is None:
            return {"detected_text": ["EMPTY"], "confidence_scores": [0.0]}

        # ---- Primary: PaddleOCR ----
        paddle_text, paddle_conf = self._run_paddle(processed)
        if paddle_text and paddle_conf >= 0.65:
            print(f"[PaddleOCR] Raw: {paddle_text} (Conf={paddle_conf:.2f})")
            return {"detected_text": [paddle_text], "confidence_scores": [paddle_conf]}

        # ---- Fallback: EasyOCR ----
        print("[WARN] PaddleOCR returned low confidence — switching to EasyOCR fallback")
        easy_text, easy_conf = self._run_easyocr(processed)
        return {"detected_text": [easy_text], "confidence_scores": [easy_conf]}

    # -----------------------------------------------------------
    # PaddleOCR inference
    # -----------------------------------------------------------
    def _run_paddle(self, img):
        try:
            results = self.paddle.ocr(img, cls=True)
            texts, confs = [], []

            for line in results[0]:
                txt, conf = line[1][0].upper(), line[1][1]
                txt = re.sub(r"[^A-Z0-9]", "", txt)
                if txt:
                    texts.append(txt)
                    confs.append(conf)

            if not texts:
                return "", 0.0

            combined = "".join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return combined, avg_conf
        except Exception as e:
            print(f"[ERROR] PaddleOCR failed → {e}")
            return "", 0.0

    # -----------------------------------------------------------
    # EasyOCR fallback
    # -----------------------------------------------------------
    def _run_easyocr(self, img):
        try:
            results = self.easy.readtext(img)
            if not results:
                return "NO_TEXT", 0.0

            texts = [re.sub(r"[^A-Z0-9]", "", t[1].upper()) for t in results]
            combined = "".join(texts)
            avg_conf = float(np.mean([t[2] for t in results])) if results else 0.0
            print(f"[EasyOCR] Raw: {combined} (Conf={avg_conf:.2f})")
            return combined, avg_conf
        except Exception as e:
            print(f"[ERROR] EasyOCR failed → {e}")
            return "ERROR", 0.0
