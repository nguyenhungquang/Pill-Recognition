## Source code for AI4VN 2022 - VAIPE: Medicine Pill Image Recognition Challenge

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