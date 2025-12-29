from PIL import Image
import torchvision.utils as vutils

def load_image(path, preprocess, device):
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)

def save_image(tensor, path):
    vutils.save_image(tensor, path)
