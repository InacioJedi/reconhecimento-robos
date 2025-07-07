#!/usr/bin/env python3

import os
from ultralytics import YOLO
import cv2

# ================================
# Script: train_detector.py
# ================================
# - Treina o modelo YOLOv8 com robos.yaml
# - Permite inferência em imagens, vídeos ou webcam com streaming para reduzir uso de memória
# ================================


def train_model(data_yaml: str,
                model_name: str = 'yolov8n.pt',
                epochs: int = 50,
                imgsz: int = 640,
                batch: int = 16,
                run_name: str = 'robo_detector'):
    """
    Treina um detector YOLOv8.
    """
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name
    )
    print(f"✔ Treino concluído. Pesos em runs/detect/{run_name}/weights/best.pt")


def run_inference(source: str,
                  weights: str,
                  show: bool = True,
                  save: bool = False):
    """
    Realiza inferência em imagens, vídeos ou webcam.

    - source: arquivo (jpg/png/mp4), diretório de imagens ou webcam ('0')
    - weights: caminho para .pt (best.pt)
    - show: exibe janelas
    - save: salva resultados em runs/detect/predict
    """
    args = dict(model=weights, source=source, stream=True)
    if show:
        args['show'] = True
    if save:
        args['save'] = True

    ext = os.path.splitext(source)[1].lower()
    if ext in ['.mp4', '.avi', '']:
        # track mantém IDs entre frames e usa streaming
        YOLO(weights).track(**args)
    else:
        YOLO(weights).predict(**args)


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(base, 'robos.yaml')

    # 1. Treine o modelo (descomente se quiser treinar aqui)
    # train_model(data_yaml)

    # 2. Inferência:
    weights = os.path.join(base, 'runs', 'detect', 'robo_detector', 'weights', 'best.pt')

    # Exemplo: processamento de vídeo com caixas exibidas e resultados salvos
    video_path = os.path.join(base, 'data', 'images', 'videos', 'video 1.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vídeo não encontrado em {video_path}")
    run_inference(source=video_path, weights=weights, show=True, save=True)

    # Outros exemplos:
    # processa imagem única:
    # run_inference(source=os.path.join(base, 'data', 'images', 'val', 'video 1_00000.jpg'), weights=weights)

    # processa todos frames de validação:
    # run_inference(source=os.path.join(base, 'data', 'images', 'val'), weights=weights, save=True)

    # webcam:
    # run_inference(source='0', weights=weights)
