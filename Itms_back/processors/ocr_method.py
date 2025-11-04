import cv2
import numpy as np
import easyocr
import re
import os
from collections import defaultdict

class SimpleOCRProcessor:
    def __init__(self, lang="en", use_gpu=True, smooth_window=5):
        """
        OCR engine for Indian license plates using EasyOCR + temporal smoothing.
        """
        self.reader = easyocr.Reader([lang], gpu=use_gpu)
        self.smooth_window = smooth_window  # how many frames to keep per vehicle
        self.ocr_history = defaultdict(list)  # {vehicle_id: [(text, conf)]}
        print(f"[INFO] EasyOCR initialized (GPU={use_gpu}, smoothing={smooth_window})")

    # ---------------------------------------------------------
    # Preprocessing: upscale small plates, convert to grayscale
    # ---------------------------------------------------------
    def _prepare(self, img):
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        if h < 150:
            scale = 200 / h
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
        os.makedirs("debug", exist_ok=True)
        cv2.imwrite("debug/ocr_input.jpg", gray)
        return gray

    # ---------------------------------------------------------
    # Main OCR + smoothing
    # ---------------------------------------------------------
    def extract_text_from_image(self, img, vehicle_id=None):
        if img is None or img.size == 0:
            return {"detected_text": ["EMPTY"], "confidence_scores": [0.0]}

        try:
            processed = self._prepare(img)
            if processed is None:
                return {"detected_text": ["EMPTY"], "confidence_scores": [0.0]}

            results = self.reader.readtext(processed)
            if not results:
                # fallback with threshold
                thresh = cv2.adaptiveThreshold(processed, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 17, 9)
                results = self.reader.readtext(thresh)
                if not results:
                    return {"detected_text": ["NO_TEXT"], "confidence_scores": [1.0]}

            texts = [re.sub("[^A-Z0-9]", "", t[1].upper()) for t in results]
            combined = "".join(texts)
            avg_conf = float(np.mean([t[2] for t in results])) if results else 1.0

            corrected = self._correct_indian_plate(combined)
            print(f"[EASY OCR] Raw: {combined} → Corrected: {corrected} (Conf={avg_conf:.2f})")

            # Save to smoothing buffer if vehicle_id given
            if vehicle_id is not None:
                self._update_ocr_history(vehicle_id, corrected, avg_conf)
                stable_text = self._get_stable_text(vehicle_id)
                print(f"[STABLE OCR] {vehicle_id} → {stable_text}")
                return {"detected_text": [stable_text], "confidence_scores": [avg_conf]}
            else:
                return {"detected_text": [corrected], "confidence_scores": [avg_conf]}

        except Exception as e:
            print(f"[ERROR] EasyOCR exception → {e}")
            return {"detected_text": ["ERROR"], "confidence_scores": [0.0]}

    # ---------------------------------------------------------
    # OCR temporal smoothing
    # ---------------------------------------------------------
    def _update_ocr_history(self, vid, text, conf):
        """
        Keep a moving window of OCR readings for each vehicle ID.
        """
        self.ocr_history[vid].append((text, conf))
        if len(self.ocr_history[vid]) > self.smooth_window:
            self.ocr_history[vid].pop(0)

    def _get_stable_text(self, vid):
        """
        Pick the most reliable OCR string from recent history:
        - longest valid match
        - highest confidence
        - valid Indian format
        """
        entries = self.ocr_history.get(vid, [])
        if not entries:
            return "NO_TEXT"

        # Prefer valid Indian plate pattern
        valid_entries = [(t, c) for (t, c) in entries if re.match(r"[A-Z]{2}\d{1,2}[A-Z]{0,2}\d{3,4}", t)]
        if valid_entries:
            longest = max(valid_entries, key=lambda x: (len(x[0]), x[1]))
            return longest[0]

        # Fallback: longest + highest confidence
        longest = max(entries, key=lambda x: (len(x[0]), x[1]))
        return longest[0]

    # ---------------------------------------------------------
    # Correction for Indian plate patterns
    # ---------------------------------------------------------
    def _correct_indian_plate(self, text: str) -> str:
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)

        if not text:
            return "NO_TEXT"

        # Shape-based confusion map
        confusion_map = {
            "6": "G",
            "0": "O",
            "1": "I",
            "2": "Z",
            "5": "S",
            "8": "B",
            "9": "P",
            "B": "8",
            "O": "0",
            "Z": "2",
            "S": "5",
            "G": "6"
        }

        corrected = "".join(confusion_map.get(ch, ch) for ch in text)

        # Ensure first part is letters
        if re.match(r"^\d", corrected):
            corrected = corrected.replace("6", "G").replace("0", "O").replace("1", "I").replace("8", "B")

        # Match valid Indian patterns
        match = re.search(r"[A-Z]{2}\d{1,2}[A-Z]{0,2}\d{3,4}", corrected)
        if match:
            corrected = match.group()

        # Fallback to longest alphanumeric run
        if len(corrected) < 6:
            fallback = re.findall(r"[A-Z0-9]+", corrected)
            corrected = max(fallback, key=len) if fallback else "NO_TEXT"

        return corrected
