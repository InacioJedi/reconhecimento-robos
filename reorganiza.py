import os
import shutil

# ajuste estes caminhos conforme sua estrutura
BASE = r'C:\Users\inaci\TCC COMBATE\data'
SRC = os.path.join(BASE, 'labels')        # aqui estão hoje imagens e .txt misturados
IMG_DST = os.path.join(BASE, 'images', 'train')
LBL_DST = os.path.join(BASE, 'labels', 'train')

os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

for fn in os.listdir(SRC):
    src_path = os.path.join(SRC, fn)
    name, ext = os.path.splitext(fn.lower())
    if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        shutil.move(src_path, os.path.join(IMG_DST, fn))
    elif ext == '.txt':
        shutil.move(src_path, os.path.join(LBL_DST, fn))

print("Movido todas as imagens para images/train e labels para labels/train")
