import torch
import torch.nn.functional as F

def clip_targeted_loss(image_emb, target_text_emb):
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    target_text_emb = target_text_emb / target_text_emb.norm(dim=-1, keepdim=True)
    return -torch.sum(image_emb * target_text_emb)
