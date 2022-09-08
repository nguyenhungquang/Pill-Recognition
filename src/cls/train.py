# %%
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from albumentations.pytorch import ToTensorV2
import random
import os
import matplotlib.pyplot as plt
import re
import json

from torchmetrics import Accuracy, Recall, Precision
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model import *
from loss import *
from dataset import *
from utils import *

import csv
from collections import defaultdict
from copy import deepcopy

device = torch.device('cuda:1')
# %%
imgsize = 128 # img size 300 works better on validate set but hasnt been tested on test set
hid_dim = 256

train_transform = A.Compose(
    [
        # A.PadIfNeeded(min_height=imgsize, min_width=imgsize, border_mode=0, value=(0,0,0)),
        A.LongestMaxSize(max_size=imgsize, interpolation=1),
        A.PadIfNeeded(min_height=imgsize, min_width=imgsize, border_mode=0, value=(0,0,0)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.augmentations.geometric.rotate.Rotate(p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.transforms.GaussNoise(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=imgsize, interpolation=1),
        A.PadIfNeeded(min_height=imgsize, min_width=imgsize, border_mode=0, value=(0,0,0)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

with open('public_train_diagnose.json', 'r') as fr:
    train_diag = json.load(fr)

base = '../VAIPE/public_train/'
with open(base + 'pill_pres_map.json', 'r') as fr: 
    mapping = json.load(fr)
pres_list = os.listdir(base + 'prescription/label')
def get_desc(content):
    out = {'drugs':[]}
    for text in content:
        if 'drug' in text['label']:
            out['drugs'].append([re.sub(r'^\d\) ', '', text['text']), text['mapping']])
        # if 'diag' in text['label']:
        #     out['diagnose'] = text['text']
    return out
desc = {}
for f in pres_list:
    if f[-4:] == 'json':
        with open(base + 'prescription/label/' + f, 'r') as fr:
            desc[f] = get_desc(json.load(fr))
            desc[f]['diagnose'] = train_diag[os.path.splitext(f)[0]]

# map each pill to prescription
pill_pres_dict = {}
for pres in mapping:
    for pill in pres['pill']:
        assert pill not in pill_pres_dict.keys()
        pill_pres_dict[pill] = pres['pres']

pill_desc = {p: {'desc': desc[pill_pres_dict[p]]} for p in pill_pres_dict.keys()}

img_desc = []
for i in range(108):
    for f in os.listdir(f'pill_boxes/{i}'):
        pill_base = re.sub(r'_\d+.jpg', '', f) + '.json'
        img_desc.append({'path': f, 'desc': pill_desc[pill_base]['desc'], 'label': i})

config = AutoConfig.from_pretrained('xlm-roberta-base')
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

with open('drug2id.json', 'r') as fr:
    drug_list = list(json.load(fr).keys())

train_dataset = TrainDataset(img_desc,
                            train_transform, tokenizer)

val_det_path = '../yolov5/runs/detect/val_pill/labels/'
with open('../VAIPE/public_val/pill_pres_map.json', 'r') as fr:
    val_pres2img = json.load(fr)
val_img2pres = {}
for pres in val_pres2img:
    for img in pres['pill']:
        val_img2pres[os.path.splitext(img)[0]] = os.path.splitext(pres['pres'])[0]

with open('val_ocr.csv', 'r') as fr:
    ocr = csv.reader(fr, delimiter=',')
    ocr = [r for r in ocr]
ocr = [[c.replace('#', ',') for c in r if c != ''] for r in ocr]
val_pres2drugs = {}
for r in ocr:
    val_pres2drugs[os.path.splitext(r[0])[0]] = r[1:]
val_img2drug_list = {}
for k, v in val_img2pres.items():
    val_img2drug_list[os.path.splitext(k)[0]] = val_pres2drugs[os.path.splitext(v)[0]]

with open('drug2id.json', 'r') as fr:
    drug2id = json.load(fr)
val_img_filter = {val_img2pres[k]: sum([drug2id[d] for d in v], []) for k, v in val_img2drug_list.items()}

val_dataset = ValDataset(val_transform, tokenizer, val_det_path, val_img2drug_list)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True, collate_fn=train_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)


# validate
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# def warn(*args, **kwargs):
#     pass
# warnings.warn = warn
def IoU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def read_gt(file):
    label_path = '../VAIPE/public_val/pill/label'
    with open(os.path.join(label_path, file), 'r') as fr:
        gt = json.load(fr)
    return [((l['x'], l['y'], l['x'] + l['w'], l['y'] + l['h']), l['label']) for l in gt]

def find_label(pred_box, gt_boxes):
    label = -1
    max_iou = 0.5
    for box, l in gt_boxes:
        iou = IoU(pred_box, box)
        if iou > max_iou:
            label = l
        max_iou = iou
    return label

def iou_acc(box_files, box_preds):
    total = 0
    acc = 0
    preds = []
    targets = []
    for fn, pred in zip(box_files, box_preds):
        gt_boxes = read_gt(os.path.splitext(fn)[0] + '.json')
        gt_label = find_label(pred[0], gt_boxes)
        if gt_label >= 0:
            targets.append(gt_label)
            preds.append(pred[1])

    reports = precision_recall_fscore_support(targets, preds, average=None, zero_division=0, labels=range(108))
    # print(reports, len(reports), len(reports[0]))
    p, r, f, supp = reports
    weight = np.array([1] * 107 + [5]) * (supp != 0)
    p = np.average(p, weights=weight)
    r = np.average(r, weights=weight)
    f = 2 * p * r / (p + r)
    return p, r, f
    
@torch.no_grad()
def get_results(model, loader, thres=0., write=False, verbase=False):
    img_files = []
    model.eval()
    preds = []
    num_pills = []
    pres_info = []
    total_num_drugs = []
    with torch.no_grad():
        it = tqdm(loader) if verbase else loader
        for batch in loader:
            img_files += batch[0]
            num_pills += batch[2]
            img = batch[1].to(device)
            drug_ids = batch[3].to(device)
            drug_mask = batch[4].to(device)
            num_drugs = batch[5]
            diagnose_ids = batch[6].to(device)
            diagnose_mask = batch[7].to(device)
            # print(diagnose_ids.shape, drug_ids.shape)
            # print(drug_ids.shape, sum(batch[5]))
            pred, pres_pred, _ = model(img, drug_ids, drug_mask, num_drugs, return_drug=True)
            pres_info.append(pres_pred)
            total_num_drugs.append(num_drugs)
            preds.append(pred.argmax(-1))
    preds = torch.cat(preds).cpu().numpy()
    # pred_labels = preds.argmax(-1)
    pred = defaultdict(lambda: [])
    filter_results = defaultdict(lambda: [])

    start = 0
    if write:
        fw = open('results.csv', 'w')
        fw.write('image_name,class_id,confidence_score,x_min,y_min,x_max,y_max\n')
        
    box_files = []
    box_preds = []
    for i, file in enumerate(img_files):
        with open(val_det_path + os.path.splitext(file)[0] + '.txt', 'r') as fr:
            text = fr.readlines()
            text = [l.split() for l in text]
        length = num_pills[i]
        assert length == len(text)
        # pres = img2pres[file[:-3] + 'json'][:-4] + 'png'
        pres = val_img2pres[file[:-4]]#[:-4] + 'png'
        drug_filter = val_img_filter[pres]
        for j in range(length):
            if float(text[j][1]) > thres:
                pred[file].append([0, preds[start + j]])

                if int(preds[start + j]) not in drug_filter and int(preds[start + j]) != 107:
                    filter_results[file].append([0, 107])
                    text[j][0] = '107'
                else:
                    filter_results[file].append([0, preds[start + j]])
                    text[j][0] = str(preds[start + j])

                box_files.append(file)
                box_preds.append((list(map(int, text[j][2:])), int(text[j][0])))
                if write:
                    line = ','.join(text[j])
                    fw.write(f'{file},{line}\n')
        start += length
    if write:
        fw.close()

    return box_files, box_preds

effnet = torchvision.models.efficientnet_b3(torchvision.models.EfficientNet_B3_Weights.DEFAULT)

model = ClsModel(config.vocab_size, effnet, hid_dim=hid_dim).to(device)

optim = torch.optim.Adam(model.parameters(), lr=2e-5)


# %%
fp16_run = True
scaler = GradScaler(enabled=fp16_run)


# %%
loss_fn = nn.CrossEntropyLoss()
# loss_fn = focal_loss

best_f1 = 0
log_dir = 'ckpt'
os.makedirs(log_dir, exist_ok=True)

for i in range(100):
    model.train()
    j = 0
    subpbar = tqdm(train_loader)
    for batch in subpbar:
        
        img, labels, drug_ids, drug_mask, num_drugs, desc_ids, desc_mask, drug_labels = to_device(batch, device)
        with autocast(enabled=fp16_run):
            pred, pres_logits, img_aux_logits = model(img, drug_ids, drug_mask, num_drugs, return_drug=True)
        main_loss = loss_fn(pred, labels)
        pres_labels = label2onehot(drug_labels, num_drugs)
        pres_loss = torch.nn.functional.binary_cross_entropy_with_logits(pres_logits, pres_labels)
        # pres_loss = binary_focal_loss(pres_logits, pres_labels, convert=False)

        in_ind = labels != 107
        in_logits = img_aux_logits[in_ind]
        in_labels = labels[in_ind]
        img_aux_loss = torch.nn.functional.cross_entropy(in_logits, in_labels)

        loss = main_loss + pres_loss + img_aux_loss
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        # loss.backward()
        # optim.step()
        scaler.step(optim)
        scaler.update()

        message = f'Epoch: {i}, Loss: {loss.item():.3f}, Aux loss: {img_aux_loss.item():.3f}, Pres loss: {pres_loss.item():.3f}'
        subpbar.set_description(message)
    
    # validate
    box_files, box_preds = get_results(model, val_loader, thres=0.5, write=False)
    _, _, f = iou_acc(box_files, box_preds)
    if f > best_f1:
        best_f1 = f
        best_epoch = i

    torch.save(model.state_dict(), f'{log_dir}/classifier_{i}.pth')
    msg = f'Epoch: {i}, Best F1: {best_f1}, Cur F1: {f}, Best epoch: {best_epoch}\n'
    print(msg)
    with open(f'{log_dir}/log.txt', 'a') as f:
        f.write(msg)
    
    





