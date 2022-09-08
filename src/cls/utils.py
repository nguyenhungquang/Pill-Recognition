import torch
import torchmetrics
from tqdm import tqdm

def to_device(batch, device):
    img = batch[0].to(device)
    labels = batch[1].to(device)
    drug_ids = batch[2]['input_ids'].to(device)
    drug_mask = batch[2]['attention_mask'].to(device)
    num_drugs = batch[3]
    desc_ids = batch[4]['input_ids'].to(device)
    desc_mask = batch[4]['attention_mask'].to(device)
    # desc_ids = desc_mask = None
    drug_labels = batch[5].to(device)
    return img, labels, drug_ids, drug_mask, num_drugs, desc_ids, desc_mask, drug_labels

@torch.no_grad()
def accuracy(model, loader, device=torch.device('cpu'), weight=False, pres_info=False):
    model.eval()
    model = model.to(device)
    corr, total = 0, 0
    drug_corr = 0
    total_drugs = 0
    w_corr, w_total = 0, 0
    preds = []
    targets = []
    for batch in tqdm(loader):
        img, labels, drug_ids, drug_mask, num_drugs, desc_ids, desc_mask, drug_labels = to_device(batch, device)
        pred, drug_logits, _, _ = model(img, drug_ids, drug_mask, num_drugs, return_drug=True, pres_info=pres_info)
        total += img.shape[0]
        pred = pred.argmax(dim=1)
        preds.append(pred)
        targets.append(labels)
        out_class = pred == 107
        corr += (pred == labels).sum()
        if weight:
            w_corr = w_corr + (pred == labels).sum() + (pred[out_class] == labels[out_class]).sum() * 9
            w_total = w_total + img.shape[0] + out_class.sum() * 9
        drug_corr += (drug_logits.argmax(dim=1) == drug_labels).sum()
        total_drugs += len(drug_labels)
    
    
    out = [(corr / total).item(), (drug_corr / total_drugs).item()]
    if weight:
        out.append((w_corr / w_total).item())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    f1 = torchmetrics.functional.f1_score(preds, targets, num_classes=108, average='weighted')
    out.append(f1)
    return out

def label2onehot(labels, num_items, num_classes=107):
    bsz = len(num_items)
    targets = torch.zeros((bsz, num_classes), device=labels.device)
    labels = torch.split(labels, num_items)
    for i in range(bsz):
        targets[i, labels[i]] = 1
    return targets

import random
import albumentations.augmentations.crops.functional as F
from albumentations.core.transforms_interface import DualTransform

class RandomCropFromBorders(DualTransform):
    """Crop bbox from image randomly cut parts from borders without resize at the end
    Args:
        crop_left (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
        from left side in range [0, crop_left * width)
        crop_right (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
        from right side in range [(1 - crop_right) * width, width)
        crop_top (float): singlefloat value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
        from top side in range [0, crop_top * height)
        crop_bottom (float): single float value in (0.0, 1.0) range. Default 0.1. Image will be randomly cut
        from bottom side in range [(1 - crop_bottom) * height, height)
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        crop_left=0.1,
        crop_right=0.1,
        crop_top=0.1,
        crop_bottom=0.1,
        always_apply=False,
        p=1.0,
    ):
        super(RandomCropFromBorders, self).__init__(always_apply, p)
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        x_min = random.randint(0, int(self.crop_left * img.shape[1]))
        x_max = random.randint(max(x_min + 1, int((1 - self.crop_right) * img.shape[1])), img.shape[1])
        y_min = random.randint(0, int(self.crop_top * img.shape[0]))
        y_max = random.randint(max(y_min + 1, int((1 - self.crop_bottom) * img.shape[0])), img.shape[0])
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)

    def apply_to_mask(self, mask, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.clamping_crop(mask, x_min, y_min, x_max, y_max)

    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        rows, cols = params["rows"], params["cols"]
        return F.bbox_crop(bbox, x_min, y_min, x_max, y_max, rows, cols)

    def apply_to_keypoint(self, keypoint, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.crop_keypoint_by_coords(keypoint, crop_coords=(x_min, y_min, x_max, y_max))

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "crop_left", "crop_right", "crop_top", "crop_bottom"