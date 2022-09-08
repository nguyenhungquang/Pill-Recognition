#!/bin/bash
# pill detect
python yolov5/detect.py --weights yolov5/runs/train/v5s/weights/best.pt --imgsz 832 --source VAIPE/public_test/pill/image --save-txt  --save-conf --nosave --name v5s_832_tta --augment --exist-ok
python yolov5/detect.py --weights yolov5/runs/train/v5x6/weights/best.pt --imgsz 1280 --source VAIPE/public_test/pill/image --save-txt  --save-conf --nosave --name v5x6_1280_tta --augment --exist-ok
# ensemble bbox
python ensemble_det/main.py
# drug name box detect
python yolov5/detect.py --weights yolov5/runs/train/drug_name/weights/best.pt --imgsz 640 --source VAIPE/public_test/prescription/image --save-txt  --save-conf --nosave --name drug_name --exist-ok
# run ocr
python ocr.py
# inference
cd cls
python inference.py
cd ..