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
imgsize = 128
hid_dim = 256

train_transform = A.Compose(
    [
        # A.PadIfNeeded(min_height=imgsize, min_width=imgsize, border_mode=0, value=(0,0,0)),
        A.LongestMaxSize(max_size=imgsize, interpolation=1),
        A.PadIfNeeded(min_height=imgsize, min_width=imgsize, border_mode=0, value=(0,0,0)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.augmentations.geometric.rotate.Rotate(p=0.8),
        # A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, p=0.8),
        # A.RandomCrop(height=128, width=128),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
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

base = '../VAIPE/public_train/'
with open(base + 'pill_pres_map.json', 'r') as fr: 
    mapping = json.load(fr)
pres_list = os.listdir(base + 'prescription/label')
def get_desc(content):
    out = {'drugs':[]}
    for text in content:
        if 'drug' in text['label']:
            out['drugs'].append([re.sub(r'^\d\) ', '', text['text']), text['mapping']])
        if 'diag' in text['label']:
            out['diagnose'] = text['text']
    return out
desc = {}
for f in pres_list:
    if f[-4:] == 'json':
        with open(base + 'prescription/label/' + f, 'r') as fr:
            desc[f] = get_desc(json.load(fr))

# %%
pill_pres_dict = {}
for pres in mapping:
    for pill in pres['pill']:
        assert pill not in pill_pres_dict.keys()
        pill_pres_dict[pill] = pres['pres']

# %%
pill_desc = {p: {'desc': desc[pill_pres_dict[p]]} for p in pill_pres_dict.keys()}


# %%
img_desc = []
for i in range(108):
    for f in os.listdir(f'pill_boxes/{i}'):
        pill_base = re.sub(r'_\d+.jpg', '', f) + '.json'
        img_desc.append({'path': f, 'desc': pill_desc[pill_base]['desc'], 'label': i})

config = AutoConfig.from_pretrained('xlm-roberta-base')
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# %%
# with open('drop.txt', 'r') as fr:
#     drop = fr.readlines()
#     drop = [l[:-1] for l in drop]

# img_desc = [i for i in img_desc if i['path'] not in drop]

    
# %%
import pickle
if os.path.isfile('train.pkl'):
    with open('train.pkl', 'rb') as fr:
        train = pickle.load(fr)

    with open('val.pkl', 'rb') as fr:
        val = pickle.load(fr)
else:
    num_train_labels = 0
    num_val_labels = 0
    while num_train_labels != 108 or num_val_labels < 107:
        train, val = train_test_split(img_desc, test_size = 0.2)
        num_train_labels = set()
        for i in train:
            num_train_labels.add(i['label'])
        num_train_labels = len(num_train_labels)
        num_val_labels = set()
        for i in val:
            num_val_labels.add(i['label'])
        num_val_labels = len(num_val_labels)
    with open('train.pkl', 'wb') as fw:
        pickle.dump(train, fw)

    with open('val.pkl', 'wb') as fw:
        pickle.dump(val, fw)


# %%
with open('drug2id.json', 'r') as fr:
    drug_list = list(json.load(fr).keys())

vaipe_traindata = VAIPE_CLS_Dataset(img_desc,
                            train_transform)
vaipe_valdata = VAIPE_CLS_Dataset(val,
                            val_transform)
train_loader = DataLoader(vaipe_traindata, batch_size=64, num_workers=4, shuffle=True, pin_memory=True, collate_fn=vaipe_traindata.collate_fn)
test_loader = DataLoader(vaipe_valdata, batch_size=32, num_workers=4, pin_memory=True, collate_fn=vaipe_valdata.collate_fn)


# %%


# %%

effnet = torchvision.models.efficientnet_b3(torchvision.models.EfficientNet_B3_Weights.DEFAULT)
# effnet = torchvision.models.efficientnet_v2_m('DEFAULT')

# model = Classifier_multi_simple(pretrained, effnet, 6, hid_dim=hid_dim, self_att=drug_att, supcon=supcon, temperature=2).to(device)
model = ClsModel(config.vocab_size, effnet, 6, hid_dim=hid_dim, self_att=drug_att, supcon=supcon, temperature=2).to(device)
# model.img_model.load_state_dict(torch.load(f'multimodal_cons/img_model.pth', map_location='cpu'))
# model.text_model.load_state_dict(torch.load(f'multimodal_cons/text_model.pth', map_location='cpu'))
# model = nn.Sequential(effnet, nn.Linear(1000, 108)).to(device)

optim = torch.optim.Adam(model.parameters(), lr=5e-5)
# ema = ModelEmaV2(model)
# ema = EMA(
#     model,
#     beta = 0.9999,              # exponential moving average factor
#     update_after_step = 100,    # only after this number of .update() calls will it start updating
#     update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
# )

# model.load_state_dict(torch.load('classifier.pth'))
# optim.load_state_dict(torch.load('optim.pth'))

# %%
print('train')


# %%
fp16_run = True
scaler = GradScaler(enabled=fp16_run)


# %%
loss_fn = nn.CrossEntropyLoss()
# loss_fn = focal_loss

best_acc = 0
best_drug_acc = 0
log_dir = 'bce_no_focal_loss_no_att'
os.makedirs(log_dir, exist_ok=True)
with open(log_dir + '/change.txt', 'a') as fw:
    fw.write('_multi_task :hid dim 256, img aux loss, pres loss. Drug att on main branch, before concat, all dataset. \n')
model.load_state_dict(torch.load(f'bce_no_focal_loss_no_att/classifier_multi_task_33.pth', map_location='cpu'))

# accuracy(model, test_loader, device, weight=True, pres_info=pres_info)
class_freq = []
for i in range(108):
    freq = len(os.listdir(f'/mnt/sdc/shared/VAIPE/modified_cls/{i}'))
    if freq == 0:
        class_freq.append(1)
    else:
        class_freq.append(1 / freq)
class_freq = torch.tensor(class_freq, device=device) 
class_freq = class_freq / class_freq.sum()
for i in range(34, 100):
    model.train()
    j = 0
    subpbar = tqdm(train_loader)
    for batch in subpbar:
        
        img, labels, drug_ids, drug_mask, num_drugs, desc_ids, desc_mask, drug_labels = to_device(batch, device)
        with autocast(enabled=fp16_run):
            pred, drug_logits, pres_logits, img_aux_logits = model(img, drug_ids, drug_mask, num_drugs, return_drug=True, return_feat=supcon, pres_info=pres_info)
        # drug_loss = focal_loss(drug_logits, drug_labels)
        drug_loss = torch.nn.functional.binary_cross_entropy_with_logits(drug_logits, label2onehot(drug_labels, [1] * drug_labels.size(0)))
        main_loss = loss_fn(pred, labels)
        pres_labels = label2onehot(drug_labels, num_drugs)
        pres_loss = torch.nn.functional.binary_cross_entropy_with_logits(pres_logits, pres_labels)
        # pres_loss = binary_focal_loss(pres_logits, pres_labels, convert=False)

        in_ind = labels != 107
        in_logits = img_aux_logits[in_ind]
        in_labels = labels[in_ind]
        img_aux_loss = torch.nn.functional.cross_entropy(in_logits, in_labels)
        # con_loss = supcon_loss(feats, labels)
        loss = main_loss + pres_loss + img_aux_loss #+ pres_loss
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        # loss.backward()
        # optim.step()
        scaler.step(optim)
        scaler.update()
        # ema.update()
        # if j % 1000 == 0:
        #     acc = accuracy(model, test_loader)
        message = f'Epoch: {i}, Loss: {loss.item():.3f}, Aux loss: {drug_loss.item():.3f}, Pres loss: {pres_loss.item():.3f}, Acc: {best_acc:.4f}, Drug acc: {best_drug_acc:.4f}'
        subpbar.set_description(message)
    
    acc, drug_acc, w_acc, f1 = accuracy(model, test_loader, device, weight=True, pres_info=pres_info)
    # if (i + 1) % 5 == 0 or i > 20:
    _count(model, _loader)
    print(acc, drug_acc, w_acc, f1)
    best_drug_acc = max(drug_acc, best_drug_acc)
    torch.save(model.state_dict(), f'{log_dir}/classifier_multi_task_{i}.pth')
    if f1 > best_acc:
        
        # torch.save(optim.state_dict(), f'{log_dir}/optim.pth')
        best_acc = f1
    with open(f'{log_dir}/log.txt', 'a') as f:
        f.write(f'Epoch: {i}, Acc: {acc:.4f}, Drug acc: {drug_acc:.4f}, Best acc: {best_acc:.4f}, Best drug acc: {best_drug_acc:.4f}\n')
    # if (i + 1) % 10 == 0 or i == 0:
    
    





