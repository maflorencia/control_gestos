"""
------
App Gradio para HuggingFace Spaces.


Gestos:
  ☝ Índice en zona DERECHA  → Slide siguiente
  ☝ Índice en zona IZQUIERDA → Slide anterior
  ✊ Puño cerrado             → Pausar / reanudar
  🖐 Mano abierta             → Volver al inicio (slide 1)
"""

import cv2
import numpy as np
import gradio as gr

from pipeline_gestos import GestureEngine
from slide_controlador import SlideController

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
engine = GestureEngine(hold_threshold=6)
controller = SlideController(slides_dir="assets/slides")

# Cooldown: evita que un mismo gesto se dispare múltiples veces seguidas
_last_triggered = "NONE"
_cooldown_count = 0
COOLDOWN_FRAMES = 20  # frames de gracia tras disparar un gesto


# ---------------------------------------------------------------------------
# Función de procesamiento (se llama por cada frame del stream)
# ---------------------------------------------------------------------------
def process_frame(frame: np.ndarray):
    global _last_triggered, _cooldown_count

    if frame is None:
        blank = np.ones((500, 900, 3), dtype=np.uint8) * 30
        cv2.putText(blank, "Esperando cámara...", (200, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        return blank, blank, "Sin señal de cámara", "–"

    # Procesar frame con MediaPipe
    triggered, pointer, annotated_cam = engine.process_frame(frame)

    # Gestionar cooldown
    if _cooldown_count > 0:
        _cooldown_count -= 1
        triggered = "HOLD"  # ignorar durante cooldown

    feedback_msg = None

    # Disparar acción si hay gesto válido y no es el mismo que el anterior
    if triggered not in ("NONE", "HOLD"):
        feedback_msg = controller.apply_gesture(triggered)
        _last_triggered = triggered
        _cooldown_count = COOLDOWN_FRAMES

    # ---------- Renderizar slide actual ----------
    slide_bgr = controller.current_slide_bgr()
    slide_display = cv2.resize(slide_bgr, (960, 540))

    # Overlay: progreso
    overlay_bar = np.zeros((50, 960, 3), dtype=np.uint8)
    progress_text = controller.progress
    cv2.putText(overlay_bar, progress_text, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Barra de progreso
    bar_w = int((controller.index + 1) / controller.total * 960)
    cv2.rectangle(overlay_bar, (0, 45), (bar_w, 50), (80, 200, 120), -1)

    slide_display = np.vstack([slide_display, overlay_bar])

    # Overlay: puntero si está en modo POINTER
    if pointer and triggered == "HOLD" and engine._prev_gesture == "POINTER":
        h_cam, w_cam = frame.shape[:2]
        px = int(pointer[0] / w_cam * 960)
        py = int(pointer[1] / h_cam * 540)
        cv2.circle(slide_display, (px, py), 18, (50, 100, 255), -1)
        cv2.circle(slide_display, (px, py), 22, (255, 255, 255), 2)

    # Overlay: mensaje de feedback en la slide
    if feedback_msg:
        cv2.putText(slide_display, feedback_msg, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 220, 120), 2)

    # ---------- Overlay en la cámara: gesto detectado ----------
    gesture_label = engine._prev_gesture if triggered == "HOLD" else triggered
    color = {
        "NEXT": (80, 210, 120),
        "PREV": (255, 130, 80),
        "PAUSE": (80, 150, 255),
        "MENU": (255, 220, 50),
        "POINTER": (200, 200, 200),
    }.get(gesture_label, (160, 160, 160))

    cv2.putText(annotated_cam, f"Gesto: {gesture_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convertir a RGB para Gradio
    slide_rgb = cv2.cvtColor(slide_display, cv2.COLOR_BGR2RGB)
    cam_rgb = cv2.cvtColor(annotated_cam, cv2.COLOR_BGR2RGB)

    return slide_rgb, cam_rgb, controller.progress, feedback_msg or "–"


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------
DESCRIPTION = """
## 🖐️ Control de Presentaciones por Gestos — MediaPipe Hands

Usá gestos frente a la cámara para navegar la presentación:

| Gesto | Acción |
|---|---|
| 🖐️ Mano abierta (4 o 5 dedos) | ➡ Slide siguiente |
| ✊ Puño cerrado | ⬅ Slide anterior |
| ☝️ Índice solo hacia arriba | 🔴 Puntero láser |
"""

with gr.Blocks(title="Gesture Presenter", theme=gr.themes.Ocean()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            slide_output = gr.Image(
                label="📽 Presentación",
                show_label=True,
            )
        with gr.Column(scale=2):
            cam_output = gr.Image(
                label="📷 Cámara (con landmarks)",
                show_label=True,
            )

    with gr.Row():
        progress_box = gr.Textbox(label="Progreso", interactive=False, scale=2)
        feedback_box = gr.Textbox(label="Último gesto", interactive=False, scale=2)

    with gr.Row():
        webcam_input = gr.Image(
            sources=["webcam"],
            streaming=True,
            label="🎥 Activá la cámara aquí",
            visible=True,
        )

    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input],
        outputs=[slide_output, cam_output, progress_box, feedback_box],
        stream_every=0.05,  # ~20 FPS
    )

    gr.Markdown(
        "_Proyecto: MediaPipe Landmark — Control de Gestos_ | "
        "Construido con [MediaPipe](https://mediapipe.dev) + [Gradio](https://gradio.app)"
    )


if __name__ == "__main__":
    demo.launch()