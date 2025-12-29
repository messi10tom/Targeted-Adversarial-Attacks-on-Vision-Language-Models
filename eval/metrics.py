import torch
import clip

@torch.no_grad()
def clip_probs(model, image, texts, device):
    tokens = clip.tokenize(texts).to(device)
    logits, _ = model(image, tokens)
    return logits.softmax(dim=-1).squeeze(0)

def targeted_success(model, adv_img, orig_label, target_label, device):
    probs = clip_probs(
        model,
        adv_img,
        [orig_label, target_label],
        device
    )
    return probs[1] > probs[0], probs
