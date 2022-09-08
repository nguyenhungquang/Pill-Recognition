import os
import pickle
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, data, transform, tokenizer=None):
        self.base_path = 'pill_boxes'
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]

        img_path = '{}/{}/{}'.format(self.base_path, datum['label'], datum['path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        drug_list = datum['desc']['drugs']
        label = datum['label']
        num_drugs = len(datum['desc']['drugs'])
        if label != 107:
            for i in range(num_drugs):
                if drug_list[i][1] == label:
                    ind = i
            drug_names = [drug_list[ind][0]] + [drug_list[i][0] for i in range(num_drugs) if i != ind]
            drug_labels = [drug_list[ind][1]] + [drug_list[i][1] for i in range(num_drugs) if i != ind]
        else:
            drug_names = [drug[0] for drug in datum['desc']['drugs']]
            drug_labels = [drug[1] for drug in datum['desc']['drugs']]

        # diagnose = '. '.join(datum['desc']['diagnose'])
        diagnose = datum['desc']['diagnose'][0]
        return img, label, drug_names, diagnose, drug_labels

    def collate_fn(self, batch):
        img, labels, drug_names, desc, drug_labels = list(zip(*batch))
        num_drugs = [len(drugs) for drugs in drug_names]
        drug_names = sum(drug_names, [])
        drug_labels = sum(drug_labels, [])
        desc = self.tokenizer(list(desc), padding=True, return_token_type_ids=True, return_tensors='pt')
        drugs = self.tokenizer(drug_names, padding=True, return_tensors='pt')
        img = torch.stack(img, dim=0)
        labels = torch.tensor(labels)
        drug_labels = torch.tensor(drug_labels)
        return img, labels, drugs, num_drugs, desc, drug_labels

class TestDataset(Dataset):
    def __init__(self, transform, tokenizer, det_path, img2drug_list):
        self.det_path = det_path
        self.tokenizer = tokenizer
        self.data_path = '../VAIPE/public_test/pill/image/'
        self.pres_list = [img2drug_list[os.path.splitext(img)[0]] for img in os.listdir(self.data_path)]
        self.data = os.listdir(self.data_path)
        self.transform = transform
        self.cache_dir = 'cache/test'
        os.makedirs(self.cache_dir, exist_ok=True)

    def __getitem__(self, idx):

        try:
            with open(f'{self.cache_dir}/{idx}.pkl', 'rb') as fr:
                pill_img = pickle.load(fr)
        except:
            img = cv2.imread(self.data_path + self.data[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with open(self.det_path + os.path.splitext(self.data[idx])[0] + '.txt', 'r') as fr:
                pills = fr.read().split()
            pills = np.array(pills).reshape(-1, 6)[:, 2:].astype(int)
            pill_img = [img[coor[1]:coor[3], coor[0]:coor[2]] for coor in pills]
            
            with open(f'{self.cache_dir}/{idx}.pkl', 'wb') as fw:
                pickle.dump(pill_img, fw)
        if self.transform:
            pill_img = [self.transform(image=p)['image'] for p in pill_img]
        num_drugs = [len(self.pres_list[idx])] * len(pill_img)
        return self.data[idx], torch.stack(pill_img), self.pres_list[idx] * len(pill_img), num_drugs
    
    def __len__(self):
        return len(self.data)
        
    def collate_fn(self, batch):
        img_files, pill_img, pres_list, num_drugs = list(zip(*batch))
        
        drug_names = sum(pres_list, [])
        num_drugs = sum(num_drugs, [])
        tokenized_drug_names = self.tokenizer(drug_names, padding=True, return_tensors='pt')
        drug_ids = tokenized_drug_names['input_ids']
        drug_mask = tokenized_drug_names['attention_mask']
        num_pills = [len(p) for p in pill_img]
        pill_img = torch.cat(pill_img)
        return img_files, pill_img, num_pills, drug_ids, drug_mask, num_drugs