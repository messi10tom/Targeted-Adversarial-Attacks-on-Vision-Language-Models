import clip
import torch
from attacks.base import Attack
from utils.loss import clip_targeted_loss

class TextTargetedPGD(Attack):
    def __init__(self, eps, alpha, steps):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def run(self, clip_model, img, target_text):
        delta = torch.zeros_like(img, requires_grad=True)

        tokens = clip.tokenize([target_text]).to(img.device)
        target_emb = clip_model.encode_text(tokens).detach()

        for _ in range(self.steps):
            adv = (img + delta).clamp(0, 1)
            img_emb = clip_model.encode_image(adv)

            loss = clip_targeted_loss(img_emb, target_emb)

            grad = torch.autograd.grad(loss, delta)[0]

            delta.data = (delta + self.alpha * grad.sign()).clamp(-self.eps, self.eps)
            grad.zero_()
            delta = delta.detach().requires_grad_()

        return (img + delta).clamp(0, 1)
