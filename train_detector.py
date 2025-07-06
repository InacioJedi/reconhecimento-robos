from ultralytics import YOLO
import os

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(base, 'robos.yaml')
    model = YOLO('yolov8n.pt')  # troque para yolov8s.pt, m, l, etc.
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        name='robo_detector'
    )
    print("Treino finalizado. Pesos em runs/train/robo_detector/weights/")
