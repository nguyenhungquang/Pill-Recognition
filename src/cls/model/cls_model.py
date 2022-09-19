import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from .text_model import TextEncoder

class ClsModel(nn.Module):
    def __init__(self, n_vocabs, pretrained_img, hid_dim=128):
        super().__init__()
        # hid_dim = 128
        self.hid_dim = hid_dim
        self.img_model = pretrained_img
        dropout = 0.3
        self.img_ln = nn.Sequential(nn.Linear(1000, hid_dim),
                                    nn.Dropout(dropout)
                                    )
        self.text_model = TextEncoder(n_vocabs, hid_dim * 2)

        self.drug_ln = nn.Sequential(nn.Linear(hid_dim * 2, hid_dim),
                                    nn.Dropout(dropout)
                                    # nn.ReLU()
                                    )
        self.pres_classifier = nn.Linear(hid_dim, 107)
        self.img_aux_ln = nn.Linear(hid_dim, 107)
        self.classifier = nn.Sequential(
                                            nn.Linear(hid_dim * 2, hid_dim * 2),
                                            nn.Tanh(),
                                            nn.Dropout(dropout),
                                            nn.Linear(hid_dim * 2, 108)
                                        )

    def padding(self, drugs, num_drugs):
        mask = torch.ones(drugs.shape[0], device=drugs.device)
        drugs = drugs.split(num_drugs)
        mask = mask.split(num_drugs)
        drugs = torch.nn.utils.rnn.pad_sequence(drugs, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        return drugs, mask

    def forward(self, img, drug_ids, drug_mask, num_drugs, return_drug=False):
        img = self.img_model(img)

        img = self.img_ln(img)

        drugs = self.text_model(drug_ids, attention_mask=drug_mask)
        drugs = self.drug_ln(drugs)
        
        drugs, mask = self.padding(drugs, num_drugs)
        pres = drugs.sum(dim=1)

        feat = torch.cat([img, pres], dim=-1)
        out = self.classifier(feat)
        if not return_drug:
            return out
        pres_labels = self.pres_classifier(pres)
        return out, pres_labels, self.img_aux_ln(img)