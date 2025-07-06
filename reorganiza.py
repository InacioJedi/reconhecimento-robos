#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

def reorganize(data_root: Path, val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    src = data_root / "labels"         # onde estão hoje .jpg + .txt misturados
    img_train = data_root / "images" / "train"
    lbl_train = data_root / "labels" / "train"
    img_val   = data_root / "images" / "val"
    lbl_val   = data_root / "labels" / "val"

    # Cria as pastas, se não existirem
    for d in (img_train, lbl_train, img_val, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    # Lista todos os arquivos .txt (cada um corresponde a uma imagem)
    txt_files = list(src.glob("*.txt"))
    stems = [p.stem for p in txt_files]
    random.shuffle(stems)

    n_val = int(len(stems) * val_ratio)
    val_set = set(stems[:n_val])

    for stem in stems:
        txt_src = src / f"{stem}.txt"
        # procura a imagem correspondente em src
        img_src = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = src / f"{stem}{ext}"
            if candidate.exists():
                img_src = candidate
                break
        if img_src is None:
            print(f"[!] Não achei imagem para '{stem}', pulando.")
            continue

        # decide destino train vs val
        if stem in val_set:
            img_dst = img_val / img_src.name
            txt_dst = lbl_val / txt_src.name
        else:
            img_dst = img_train / img_src.name
            txt_dst = lbl_train / txt_src.name

        shutil.move(str(img_src), str(img_dst))
        shutil.move(str(txt_src), str(txt_dst))

    print(f"Dataset reorganizado em:\n"
          f"  train -> {len(stems)-n_val} pares\n"
          f"  val   -> {n_val} pares")

if __name__ == "__main__":
    base = Path(__file__).parent.resolve()
    data_dir = base / "data"
    reorganize(data_dir)
