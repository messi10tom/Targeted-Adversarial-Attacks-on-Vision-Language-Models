import torch.nn as nn

class AnyAttackLite(nn.Module):
    def __init__(self, latent_dim, hidden_dim, img_shape, eps):
        super().__init__()
        self.eps = eps
        C, H, W = img_shape

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, C * H * W)
        )

    def forward(self, z):
        delta = self.net(z)
        delta = delta.view(z.size(0), 3, 224, 224)
        return delta.clamp(-self.eps, self.eps)
