#!/usr/bin/env python3
"""
Interactive annotation tool for YOLO-format bounding boxes on collision event images.
"""
import os
import argparse
import cv2


def normalize_bbox(x1, y1, x2, y2, w, h):
    """
    Convert corner bounding box coordinates to YOLO format: (cx, cy, bw, bh), all normalized [0,1].
    """
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def annotate(img_dir: str, label_dir: str, class_id: int = 0):
    """
    Launch a window to draw bounding boxes with the mouse. Save annotations in YOLO txt files.
    -- Press 's' to save and go to next image.
    -- Press 'n' to skip without saving.
    -- Press 'q' to exit.
    """
    os.makedirs(label_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    for img_name in files:
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠ Could not read '{img_name}', skipping.")
            continue

        h, w = image.shape[:2]
        display = image.copy()
        bboxes = []
        drawing = False
        ix = iy = -1

        def mouse_callback(event, x, y, flags, param):
            nonlocal ix, iy, drawing, display
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                display = image.copy()
                cv2.rectangle(display, (ix, iy), (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1, x2, y2 = min(ix, x), min(iy, y), max(ix, x), max(iy, y)
                bboxes.append((x1, y1, x2, y2))
                display = image.copy()
                for bb in bboxes:
                    cv2.rectangle(display, bb[:2], bb[2:], (0, 255, 0), 2)

        cv2.namedWindow('Annotate', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Annotate', mouse_callback)

        print(f"Annotating '{img_name}'. {len(bboxes)} boxes so far.")
        while True:
            cv2.imshow('Annotate', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # save labels
                label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
                with open(label_path, 'w') as f:
                    for bb in bboxes:
                        cx, cy, bw, bh = normalize_bbox(*bb, w, h)
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                print(f"[S] Saved to '{label_path}'")
                break
            elif key == ord('n'):
                print(f"[N] Skipped '{img_name}'")
                break
            elif key == ord('q'):
                print("Exiting...")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()


def main():
    """Parse arguments and start annotation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_img = os.path.join(script_dir, 'data', 'images', 'train')
    default_lbl = os.path.join(script_dir, 'data', 'images', 'labels')

    parser = argparse.ArgumentParser(
        description='Annotate images for YOLO training (collision events).')
    parser.add_argument('--img-dir', type=str, default=default_img,
                        help='Directory with input images.')
    parser.add_argument('--label-dir', type=str, default=default_lbl,
                        help='Directory to save YOLO label files.')
    parser.add_argument('--class-id', type=int, default=0,
                        help='Class ID to assign to every box.')
    args = parser.parse_args()

    print(f"Images dir : {args.img_dir}")
    print(f"Labels dir : {args.label_dir}")
    annotate(args.img_dir, args.label_dir, args.class_id)


if __name__ == '__main__':
    main()
