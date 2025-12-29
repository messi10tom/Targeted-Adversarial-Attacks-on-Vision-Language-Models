import yaml
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.clip_wrapper import CLIPWrapper
from train.train_anyattack import train_anyattack
import torch

cfg = yaml.safe_load(open("configs/default.yaml"))
device = cfg["model"]["device"]
clip_model = CLIPWrapper(cfg["model"]["name"], device)

dataset = ImageFolder(
    os.path.join(os.getcwd(), "data/"),
    transform=clip_model.preprocess
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)



generator = train_anyattack(
    clip_model=clip_model.model,
    dataloader=loader,
    target_text="a photo of a cat",
    device=device,
    cfg=cfg["anyattack"]
)

torch.save(generator.state_dict(), "anyattack.pt")

