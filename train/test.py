import yaml
import torch
from attacks.anyattack_lite import AnyAttackLite
from models.clip_wrapper import CLIPWrapper
from utils.image import load_image
import torchvision.utils as vutils

cfg = yaml.safe_load(open("configs/default.yaml"))
device = cfg["model"]["device"]
clip_model = CLIPWrapper(cfg["model"]["name"], device)

gen = AnyAttackLite(
    latent_dim=cfg["anyattack"]["latent_dim"],
    hidden_dim=cfg["anyattack"]["hidden_dim"],
    img_shape=(3, 224, 224),
    eps=cfg["attack"]["eps"]
).to(device)


img = load_image(
    "data/dogs/dog.jpg",
    clip_model.preprocess,
    device
)

gen.load_state_dict(torch.load("anyattack.pt"))
gen.eval()

z = torch.randn(1, cfg["anyattack"]["latent_dim"]).to(device)
delta = gen(z).view_as(img)

adv = (img + delta).clamp(0, 1)


vutils.save_image(adv, "adv.png")
