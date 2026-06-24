"""
slide_controller.py
-------------------
Gestiona el estado de la presentación: índice actual, pausa, etc.
"""

import os
import cv2
import numpy as np


class SlideController:
    def __init__(self, slides_dir: str = "assets/slides"):
        # Cargar slides ordenadas numéricamente
        files = sorted(
            [f for f in os.listdir(slides_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda x: x  # orden lexicográfico funciona con nombres tipo slide_01.png
        )
        if not files:
            raise FileNotFoundError(f"No se encontraron imágenes en '{slides_dir}'")

        self.slides = [os.path.join(slides_dir, f) for f in files]
        self.index = 0
        self.paused = False
        self.total = len(self.slides)

    # ------------------------------------------------------------------

    def current_slide_bgr(self) -> np.ndarray:
        return cv2.imread(self.slides[self.index])

    def apply_gesture(self, gesture: str) -> str | None:
        """
        Aplica el gesto al estado.
        Devuelve un mensaje de feedback o None.
        """
        if gesture == "NEXT":
            if self.index < self.total - 1:
                self.index += 1
                self.paused = False
                return "➡ Siguiente"
            else:
                return "⚠ Última slide"

        elif gesture == "PREV":
            if self.index > 0:
                self.index -= 1
                self.paused = False
                return "⬅ Anterior"
            else:
                return "⚠ Primera slide"

        elif gesture == "PAUSE":
            self.paused = not self.paused
            return "⏸ Pausado" if self.paused else "▶ Reanudado"

        elif gesture == "MENU":
            self.index = 0
            self.paused = False
            return "🏠 Menú (slide 1)"

        return None

    @property
    def progress(self) -> str:
        state = "⏸" if self.paused else "▶"
        return f"{state}  Slide {self.index + 1} / {self.total}"
