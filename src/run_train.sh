#!/bin/bash
# train detection
cd yolov5
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
python convert_ai4vn.py
python convert_drugname.py
python train.py --img 640 --batch 16 --epochs 3 --data pill.yaml --weights yolov5s.pt
python train.py --img 1280 --batch 16 --epochs 3 --data pill.yaml --weights yolov5x6.pt
python train.py --img 640 --batch 16 --epochs 3 --data drugname.yaml --weights yolov5s.pt
cd ..
# train cls
python yolov5/detect.py --weights yolov5/runs/train/v5s/weights/best.pt --imgsz 832 --source VAIPE/public_val/pill/image --save-txt  --save-conf --nosave --name val_pill --augment --exist-ok
cd cls
gdown https://drive.google.com/file/d/1OA9QC0ZHmJQHLK_z58RhWiTLBPY7TFNu/view?usp=sharing
unzip pill_bboxes.zip
python train.py