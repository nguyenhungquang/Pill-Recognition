import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2, weight=None):
    # cross entropy version
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = - (1 - probs) ** gamma * log_probs
    loss = loss.gather(1, targets.unsqueeze(1))
    if weight is not None:
        alpha = weight.gather(0, targets).unsqueeze(-1)
        loss = loss * alpha
    return loss.mean()

def binary_focal_loss(logits, labels, gamma=2, alpha=0.25, convert=True):
    # bce version
    bsz = logits.shape[0]
    if convert:
        targets = torch.zeros_like(logits)
        targets[range(bsz), labels.view(-1)] = 1
    else:
        targets = labels
    targets = targets.type_as(logits)
    logits_sigmoid = logits.sigmoid()
    pt = (1 - logits_sigmoid) * targets + logits_sigmoid * (1 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # return bce.mean()
    # pt = torch.exp(-bce)
    loss = (alpha * targets + (1 - alpha) * (1 - targets)) * pt ** gamma * bce
    return loss.mean()

def supcon_loss(feats, labels, temperature=0.1):
    bsz = len(feats)
    labels = labels.view(-1, 1)
    mask = labels == labels.T
    dot_logits = feats @ feats.T / temperature
    logits_max, _ = torch.max(dot_logits, dim=1, keepdim=True)
    logits = dot_logits - logits_max.detach()  #log sum exp
    logits_mask = torch.ones_like(mask)
    logits_mask[range(bsz), range(bsz)] = 0
    mask[range(bsz), range(bsz)] = 0
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    selected_idx = mask.sum(1) != 0
    mean_log_prob_pos = (mask * log_prob).sum(1)[selected_idx] / mask.sum(1)[selected_idx]
    loss = - mean_log_prob_pos.mean()
    return loss

def cross_cons_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 1 - (x * y).sum(dim=-1).mean()