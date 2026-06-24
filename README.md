---
title: Presentaciones por Gestos
emoji: 🖐️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🖐️ ☝️  Control de Presentaciones por Gestos

Proyecto de Computer Vision con **MediaPipe Hands** y **Gradio**.  
Navega slides usando gestos de mano frente a la cámara, sin tocar el teclado.

## Gestos disponibles

| Gesto | Acción |
|---|---|
| 🖐️ Mano abierta (4 o 5 dedos) | ➡ Slide siguiente |
| ✊ Puño cerrado | ⬅ Slide anterior |
| ☝️ Índice solo hacia arriba | 🔴 Puntero láser |

---

## Estructura del proyecto

```
presentacion_gestos/
├── app.py                  # App Gradio principal
├── pipeline_gestos.py       # Lógica de detección de gestos (MediaPipe)
├── slide_controlador.py     # Estado de la presentación
├── assets/
│   └── slides/             # Slides como imágenes PNG/JPG
├── requirements.txt
├── .gitignore
└── README.md
```

---


## Stack tecnológico

- [MediaPipe Hands](https://mediapipe.dev) — detección de landmarks de mano
- [OpenCV](https://opencv.org) — procesamiento de frames
- [Gradio](https://gradio.app) — interfaz web y streaming de webcam
- [HuggingFace Spaces](https://huggingface.co/spaces) — deploy gratuito
