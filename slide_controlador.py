"""
-------------------
Gestiona el estado de la presentación: índice actual y navegación.

Gestos soportados:
  NEXT → avanza una slide
  PREV → retrocede una slide
"""

import os
import cv2
import numpy as np


class SlideController:
    def __init__(self, slides_dir: str = "assets/slides"):
        files = sorted(
            [f for f in os.listdir(slides_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda x: x
        )
        if not files:
            raise FileNotFoundError(f"No se encontraron imágenes en '{slides_dir}'")

        self.slides = [os.path.join(slides_dir, f) for f in files]
        self.index = 0
        self.total = len(self.slides)

    def current_slide_bgr(self) -> np.ndarray:
        return cv2.imread(self.slides[self.index])

    def apply_gesture(self, gesture: str) -> str | None:
        if gesture == "NEXT":
            if self.index < self.total - 1:
                self.index += 1
                return "➡ Siguiente"
            else:
                return "⚠ Última slide"

        elif gesture == "PREV":
            if self.index > 0:
                self.index -= 1
                return "⬅ Anterior"
            else:
                return "⚠ Primera slide"

        return None

    @property
    def progress(self) -> str:
        return f"▶  Slide {self.index + 1} / {self.total}"