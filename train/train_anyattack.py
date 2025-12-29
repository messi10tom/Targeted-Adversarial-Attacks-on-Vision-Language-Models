import torch
import clip
from attacks.anyattack_lite import AnyAttackLite
from utils.loss import clip_targeted_loss

def train_anyattack(
    clip_model,
    dataloader,
    target_text,
    device,
    cfg
):
    z_dim = cfg["latent_dim"]
    model = AnyAttackLite(z_dim, cfg["hidden_dim"], (3, 224, 224), 0.031).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    text_emb = clip_model.encode_text(
        clip.tokenize([target_text]).to(device)
    ).detach()

    for img, _ in dataloader:
        img = img.to(device)
        z = torch.randn(img.size(0), z_dim).to(device)

        delta = model(z)
        adv = (img + delta).clamp(0, 1)

        img_emb = clip_model.encode_image(adv)
        loss = clip_targeted_loss(img_emb, text_emb)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model
