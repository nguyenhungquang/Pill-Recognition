## Source code for AI4VN 2022 - VAIPE: Medicine Pill Image Recognition Challenge
Updated (19/9/22): Add code used to inference private test. Since private test and public test have different format, this repo can no longer run on public test dataset. New model was trained on public val dataset also.

#### Inference (private test)

Put private test dataset in folder **`RELEASE_private_test`** in this directory <br/>
Download classifier checkpoint from [https://drive.google.com/file/d/1xKYbGMgBygSKwHom5No7RLOT_IAmxooA/view?usp=sharing](https://drive.google.com/file/d/1xKYbGMgBygSKwHom5No7RLOT_IAmxooA/view?usp=sharing) and put it in **`src/cls`** folder <br/>

```
cd src/cls
gdown 1xKYbGMgBygSKwHom5No7RLOT_IAmxooA
cd ../..
```
Download detection model from [https://drive.google.com/file/d/16LoEuM1w0PVOpkIEkiAT-6ASsWe7F4OF/view?usp=sharing](https://drive.google.com/file/d/16LoEuM1w0PVOpkIEkiAT-6ASsWe7F4OF/view?usp=sharing) and put in **`src/yolov5`** folder
```
cd src/yolov5
gdown 16LoEuM1w0PVOpkIEkiAT-6ASsWe7F4OF
unzip runs.zip
cd ../..
```
Directory will look like this
```
.
└── src/
    ├── cls/
    │   ├── classifier_29.pth
    │   └── ...
    ├── yolov5/
    │   ├── runs
    │   └── ...
    └── RELEASE_private_test/
        ├── pill
        ├── prescription
        └── pill_pres_map.json
```
To build docker image for inference, use this following command
```
docker build -t base_image .
```
To inference, run 
```
docker run \
    --name inference_private_test \
    --mount type=bind,source="$(pwd)"/RELEASE_private_test,target=/app/src/RELEASE_private_test \
    --gpus all \
    base_image \
    bash run_inference.sh
```

#### Inference (public test, deprecated)

Put all dataset in folder **`VAIPE`** in this directory <br/>
Download classifier checkpoint from [https://drive.google.com/file/d/1JgAx8EY8NE_oz5al6fxST9pHa0m3OCi7/view?usp=sharing](https://drive.google.com/file/d/1JgAx8EY8NE_oz5al6fxST9pHa0m3OCi7/view?usp=sharing) and put it in **`src/cls`** folder <br/>

```
cd src/cls
gdown 1JgAx8EY8NE_oz5al6fxST9pHa0m3OCi7
cd ../..
```
Download detection model from [https://drive.google.com/file/d/1j_FLzqysevpyBo-eps-4EidDLniwboG2/view?usp=sharing](https://drive.google.com/file/d/1j_FLzqysevpyBo-eps-4EidDLniwboG2/view?usp=sharing) and put in **`src/yolov5`** folder
```
cd src/yolov5
gdown 1j_FLzqysevpyBo-eps-4EidDLniwboG2
unzip runs.zip
cd ../..
```
Directory will look like this
```
.
├── src/
│   └── cls/
│       ├── classifier_36.pth 
│       └── ...
└── VAIPE/
    ├── public_train/
    │   ├── pill/
    │   │   └── ...
    │   ├── prescription/
    │   │   └── ...
    │   └── pill_pres_map.json
    ├── public_val
    └── public_test
```
To build docker image for inference, use this following command
```
docker build -t base_image .
```
To inference, run 
```
docker run \
    --name inference_test \
    --mount type=bind,source="$(pwd)"/VAIPE,target=/app/src/VAIPE \
    --gpus all \
    --rm \
    base_image \
    bash run_inference.sh
```

#### Training
This script will first download cropped pill bboxes (we have corrected labels of some bboxes) from (https://drive.google.com/file/d/1OA9QC0ZHmJQHLK_z58RhWiTLBPY7TFNu/view?usp=sharing)[https://drive.google.com/file/d/1OA9QC0ZHmJQHLK_z58RhWiTLBPY7TFNu/view?usp=sharing] in folder **`cls`** <br/>
To train, run 
```
docker run \
    --name inference_test \
    --mount type=bind,source="$(pwd)"/VAIPE,target=/app/src/VAIPE \
    --gpus all \
    --rm \
    base_image \
    bash run_train.sh
```

