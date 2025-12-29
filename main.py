import yaml
import torch

from models.clip_wrapper import CLIPWrapper
from attacks.text_targeted import TextTargetedPGD
from attacks.anyattack_lite import AnyAttackLite

from utils.image import load_image, save_image
from utils.seed import set_seed
from eval.metrics import targeted_success


def main():
    set_seed()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    device = cfg["model"]["device"]

    clip_model = CLIPWrapper(cfg["model"]["name"], device)

    img = load_image(
        "data/dogs/dog.jpg",
        clip_model.preprocess,
        device
    )

    original_label = "a photo of a dog"
    target_label = cfg["eval"]["target_text"]



    pgd = TextTargetedPGD(
        cfg["attack"]["eps"],
        cfg["attack"]["alpha"],
        cfg["attack"]["steps"]
    )

    adv_pgd = pgd.run(
        clip_model.model,
        img,
        target_label
    )

    pgd_success, pgd_probs = targeted_success(
        clip_model.model,
        adv_pgd,
        original_label,
        target_label,
        device
    )

    save_image(adv_pgd, "adv_pgd.png")



    any_cfg = cfg["anyattack"]

    generator = AnyAttackLite(
        latent_dim=any_cfg["latent_dim"],
        hidden_dim=any_cfg["hidden_dim"],
        img_shape=(3, 224, 224),
        eps=cfg["attack"]["eps"]
    ).to(device)

    generator.load_state_dict(torch.load("anyattack.pt", map_location=device))
    generator.eval()

    with torch.no_grad():
        z = torch.randn(1, any_cfg["latent_dim"]).to(device)
        delta = generator(z).view_as(img)
        adv_any = (img + delta).clamp(0, 1)

    any_success, any_probs = targeted_success(
        clip_model.model,
        adv_any,
        original_label,
        target_label,
        device
    )

    save_image(adv_any, "adv_anyattack.png")



    print("\n=== Targeted Attack Comparison ===\n")

    print("[PGD]")
    print(f"  Original prob : {pgd_probs[0].item():.4f}")
    print(f"  Target prob   : {pgd_probs[1].item():.4f}")
    print(f"  Success      : {bool(pgd_success)}")
    print("  Saved -> adv_pgd.png\n")

    print("[AnyAttack-lite]")
    print(f"  Original prob : {any_probs[0].item():.4f}")
    print(f"  Target prob   : {any_probs[1].item():.4f}")
    print(f"  Success      : {bool(any_success)}")
    print("  Saved -> adv_anyattack.png\n")


if __name__ == "__main__":
    main()
