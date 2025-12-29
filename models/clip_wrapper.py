import clip

class CLIPWrapper:
    def __init__(self, model_name, device):
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device

    def encode_image(self, img):
        return self.model.encode_image(img)

    def encode_text(self, text_tokens):
        return self.model.encode_text(text_tokens)

    def logits(self, img, text):
        return self.model(img, text)
