import re
import csv
import os
import pickle
from collections import defaultdict
import json
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

from transformers import AutoConfig, AutoTokenizer

from dataset import TestDataset
from model import ClsModel

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.device('cuda:0')

hid_dim = 256

data_path = '../VAIPE/public_test/pill/image'
det_path = '../ensemble_det/labels/'
with open(det_path + os.path.splitext(os.listdir(data_path)[1])[0] + '.txt', 'r') as fr:
    text = fr.readlines()
import json
with open('id2drug.json', 'r') as fr:
    id2drug = json.load(fr)
with open('drug2id.json', 'r') as fr:
    drug2id = json.load(fr)

with open('../VAIPE/public_test/pill_pres_map.json', 'r') as fr:
    pres2img = json.load(fr)

img2pres = {}
for pres in pres2img:
    for img in pres['pill']:
        img2pres[img] = pres['pres']


with open('../drug_name.csv', 'r') as fr:
    ocr = csv.reader(fr, delimiter=',')
    ocr = [r for r in ocr]
ocr = [[c.replace('#', ',') for c in r if c != ''] for r in ocr]
pres2drugs = {}
for r in ocr:
    pres2drugs[os.path.splitext(r[0])[0]] = r[1:]
img2drug_list = {}
for k, v in img2pres.items():
    img2drug_list[os.path.splitext(k)[0]] = pres2drugs[os.path.splitext(v)[0]]
img_filter = {img2pres[k]: sum([drug2id[d] for d in v], []) for k, v in img2drug_list.items()}

img_size = 128 # img size 300 works better on validate set but hasnt been tested on test set
val_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=img_size, interpolation=1),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0,0,0)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

config = AutoConfig.from_pretrained('xlm-roberta-base')
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

dataset = TestDataset(val_transform, tokenizer, det_path, img2drug_list)
loader = DataLoader(dataset, batch_size = 32, num_workers=4, collate_fn = dataset.collate_fn, pin_memory=True)

effnet = torchvision.models.efficientnet_b3()
model = ClsModel(config.vocab_size, effnet, hid_dim).to(device)
model.load_state_dict(torch.load('classifier_36.pth'))


@torch.no_grad()
def evaluate(model, loader, thres=0., write=False):
    img_files = []
    model.eval()
    preds = []
    num_pills = []
    pres_info = []
    total_num_drugs = []
    conf_score = []
    with torch.no_grad():
        for batch in loader:
            img_files += batch[0]
            num_pills += batch[2]
            img = batch[1].to(device)
            drug_ids = batch[3].to(device)
            drug_mask = batch[4].to(device)
            num_drugs = batch[5]
            pred, pres_pred, _ = model(img, drug_ids, drug_mask, num_drugs, return_drug=True)
            pres_info.append(pres_pred)
            total_num_drugs.append(num_drugs)
            preds.append(pred.argmax(-1))
            conf_score.append(pred.softmax(-1).max(-1)[0])
    preds = torch.cat(preds).cpu().numpy()
    conf_score = torch.cat(conf_score).cpu().numpy()
    pred = defaultdict(lambda: [])
    filter_results = defaultdict(lambda: [])

    start = 0
    if write:
        fw = open('results.csv', 'w')
        fw.write('image_name,class_id,confidence_score,x_min,y_min,x_max,y_max\n')
        
    for i, file in enumerate(img_files):
        with open(det_path + os.path.splitext(file)[0] + '.txt', 'r') as fr:
            text = fr.readlines()
            text = [l.split() for l in text]
        length = num_pills[i]
        assert length == len(text)
        pres = img2pres[file[:-4]]
        drug_filter = img_filter[pres]
        for j in range(length):
            if float(text[j][1]) > thres:
                pred[file].append([0, preds[start + j]])
                if int(preds[start + j]) not in drug_filter and int(preds[start + j]) != 107:
                    filter_results[file].append([0, 107])
                    text[j][0] = '107'
                else:
                    filter_results[file].append([0, preds[start + j]])
                    text[j][0] = str(preds[start + j])
                
                text[j][1] = str(conf_score[start + j])

                if write:
                    line = ','.join(text[j])
                    fw.write(f'{file},{line}\n')
        start += length
    if write:
        fw.close()

evaluate(model, loader, thres=0.5, write=True)