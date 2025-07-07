#!/usr/bin/env python3

import os
import cv2
from ultralytics import YOLO

# ================================
# Script: treino e detecção de robôs
# ================================

# Configurações do treino
DATA_YAML = 'robos.yaml'  # Assumindo na raiz do projeto
MODEL_BACKBONE = 'yolov8n.pt'
EPOCHS = 50
IMGSZ = 640
BATCH = 16
RUN_NAME = 'robo_detector'

# Configuração de detecção
CONF_THRESH = 0.25
VIDEO_PATHS = [
    os.path.join('data', 'videos', 'video 1.mp4'),
    os.path.join('data', 'images', 'videos', 'video 1.mp4')
]

def train():
    """Treina o modelo YOLOv8 no seu dataset"""
    # Use modelo maior e augmentações para melhorar aprendizado
    model = YOLO('yolov8s.pt')  # backbone maior
    print(f"Iniciando treino: epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}, augment=True")
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=IMGSZ,
        batch=BATCH,
        name=RUN_NAME,
        augment=True  # ativa mosaic, mixup e outras
    )
    print("✔ Treino concluído")(f"Iniciando treino: {EPOCHS} épocas, imgsz={IMGSZ}, batch={BATCH}")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=RUN_NAME
    )
    print("✔ Treino concluído")


def detect():
    """Executa detecção no primeiro vídeo encontrado e exibe caixas"""
    # Localiza vídeo
    video = None
    for p in VIDEO_PATHS:
        if os.path.isfile(p):
            video = p
            break
    if not video:
        raise FileNotFoundError(f"Vídeo não encontrado em nenhum dos caminhos: {VIDEO_PATHS}")
    print(f"Usando vídeo: {video}")

    # Carrega pesos do treino
    weights = os.path.join('runs', 'detect', RUN_NAME, 'weights', 'best.pt')
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Pesos não encontrados em {weights}. Rode primeiro mode 'train'.")
    model = YOLO(weights)

    # Abre vídeo
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Erro ao abrir vídeo: {video}")

    # Loop de detecção
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Detecta robôs
        results = model.predict(source=frame, conf=CONF_THRESH, stream=False)
        # Debug: conte quantas detecções por frame
        for r in results:
            num_boxes = len(r.boxes)
            print(f"Frame {frame_count}: {num_boxes} boxes detected")
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                if conf < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # If no boxes at all
        if all(len(r.boxes) == 0 for r in results):
            print(f"Frame {frame_count}: no detections")
        cv2.imshow('Detecção de Robôs', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Treino (train) ou detecção (detect)')
    parser.add_argument('mode', choices=['train', 'detect'], help='Modo de operação')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        detect()
