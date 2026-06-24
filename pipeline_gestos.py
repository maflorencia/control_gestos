"""
-----------------
Detecta gestos de mano usando MediaPipe Hands.

Gestos implementados:
  NEXT    → 🖐️ mano abierta (5 dedos extendidos)
  PREV    → ✊ puño cerrado (5 dedos doblados)
  POINTER → ☝️ índice solo hacia arriba (resto doblados)
  NONE    → ningún gesto reconocido
"""

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class GestureEngine:
    def __init__(
        self,
        hold_threshold: int = 6,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.hold_threshold = hold_threshold
        self._prev_gesture = "NONE"
        self._hold_count = 0

    def _fingers_up(self, lm) -> list[int]:
        """Devuelve [pulgar, índice, medio, anular, meñique]: 1=extendido, 0=doblado."""
        fingers = []
        # Pulgar: comparar en X
        fingers.append(1 if lm[4].x < lm[3].x else 0)
        # Resto: tip.y < pip.y → extendido
        for tip_id, pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            fingers.append(1 if lm[tip_id].y < lm[pip_id].y else 0)
        return fingers

    def _classify(self, fingers: list[int]) -> str:
        p, i, m, a, me = fingers
        total = sum(fingers)

        # 🖐️ Mano abierta: 4 o 5 dedos extendidos → NEXT
        if total >= 4:
            return "NEXT"

        # ✊ Puño cerrado: 0 o 1 dedos extendidos (el pulgar puede quedar semi-afuera)
        if total <= 1:
            return "PREV"

        # ☝️ Índice solo → POINTER
        if i == 1 and m == 0 and a == 0 and me == 0:
            return "POINTER"

        return "NONE"

    def process_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        annotated = frame.copy()
        pointer = None

        if not results.multi_hand_landmarks:
            self._prev_gesture = "NONE"
            self._hold_count = 0
            return "NONE", None, annotated

        hand_lm = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            annotated,
            hand_lm,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        lm = hand_lm.landmark
        fingers = self._fingers_up(lm)
        gesture = self._classify(fingers)

        # Coordenadas del tip del índice para el puntero
        pointer = (int(lm[8].x * w), int(lm[8].y * h))

        # Debounce
        if gesture == self._prev_gesture:
            self._hold_count += 1
        else:
            self._hold_count = 0
            self._prev_gesture = gesture

        triggered = gesture if self._hold_count >= self.hold_threshold else "HOLD"

        # Mostrar gesto detectado en el frame
        color = {
            "NEXT": (80, 210, 120),
            "PREV": (80, 130, 255),
            "POINTER": (200, 200, 200),
        }.get(self._prev_gesture, (160, 160, 160))

        cv2.putText(annotated, f"{self._prev_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return triggered, pointer, annotated