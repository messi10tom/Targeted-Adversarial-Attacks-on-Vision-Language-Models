import torch
from attacks.base import Attack

class PGDAttack(Attack):
    def __init__(self, eps, alpha, steps):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def run(self, model, img, loss_fn):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        delta = torch.zeros_like(img, requires_grad=True)

        for _ in range(self.steps):
            loss = loss_fn((img + delta).clamp(0, 1))
            grad = torch.autograd.grad(loss, delta)[0]
            loss.backward()

            with torch.no_grad():
                delta += self.alpha * grad.sign()
                delta.clamp_(-self.eps, self.eps)

            delta.grad.zero_()

        return (img + delta).clamp(0, 1)
