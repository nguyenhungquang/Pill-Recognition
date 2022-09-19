# %%
import torch
import numpy as np
import cv2
import json
from PIL import Image
from fuzzywuzzy import fuzz
import re
import os

# %%
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
config = Cfg.load_config_from_name('vgg_transformer')

# %%
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False

# %%
detector = Predictor(config)

# %%
det_path = 'yolov5/runs/detect/drug_name_test/labels/'

# %%
prescription_dir = 'RELEASE_private_test/prescription/image'
preslist = [i for i in os.listdir(prescription_dir) if i[-3:] == 'png']
preslist.sort(key=lambda x:int(x[:-4].split('_')[-1]))

# %%
det_results = {}
for f in os.listdir(det_path):
    assert not f in det_results.keys()
    with open(det_path + f, 'r') as fr:
        boxes = []
        for l in fr.readlines():
            box = l.split()[2:]
            # print(box)
            box = [int(i) for i in box]
            if l.split()[0] == '0':
                boxes.append(box)
        det_results[f[:-4]] = boxes

# %%
with open('cls/drug2id.json', 'r') as fr:
    drug_list = list(json.load(fr).keys())

# %%
def get_drug_list(filename, threshold=50):
    img = Image.open(os.path.join(prescription_dir, filename))
    texts = []
    for coor in det_results[filename[:-4]]:
        box = img.crop(coor)
        text = detector.predict(box)
        texts.append(text)
    drug_names = []
    for text in texts:
        if len(text) < 1:
            continue
        drug_name = ''
        def fuzzy_match(text):
            max_ratio = 0
            for i, drug in enumerate(drug_list):
                rt = fuzz.ratio(text, drug)
                if rt > max_ratio:
                    max_ratio = rt
                    drug_name = drug
            return drug_name, max_ratio
        drug_name, max_ratio = fuzzy_match(text)
        if max_ratio > threshold:
            drug_names.append(drug_name)

    return drug_names 

# %%
drug_dict = {pres: get_drug_list(pres) for pres in preslist}

# %%
with open('drug_name_v2.csv', 'w') as fr:
    for k, v in drug_dict.items():

        drugs = ','.join([d.replace(',', '#') for d in v])
        fr.write(f'{k},{drugs}\n')

# %%



