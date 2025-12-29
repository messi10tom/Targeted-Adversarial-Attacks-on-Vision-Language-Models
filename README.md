# Targeted Adversarial Attacks on Vision–Language Models (CLIP)

This repository implements **targeted adversarial attacks on Vision–Language Models (VLMs)**, specifically **CLIP (ViT-B/32)**, as part of *Assignment B – Targeted Adversarial Attacks on VLMs*.

The project covers:
- **Optimization-based attacks** (Text-targeted PGD)
- **Learning-based attacks** inspired by **AnyAttack**
- Comparative evaluation using **attack success rate (ASR)**

The focus is on **research-prototype correctness**, not SOTA performance.

---

## Implemented Attacks

### 1. Text-Targeted PGD (Baseline)
- Per-image, test-time optimization
- No training required
- Strong but slow
- Uses CLIP image–text similarity loss

### 2. AnyAttack-lite (Learned Generator)
- Trains a perturbation generator
- One forward pass at inference
- Inspired by *AnyAttack*, scaled down for assignment feasibility
- Demonstrates limits of small-data adversarial training

---

## Repository Structure (Simplified)

```

.
├── main.py                  # Run attacks + save adversarial images
├── configs/
│   └── default.yaml         # All hyperparameters
├── attacks/
│   ├── pgd.py               # Text-targeted PGD
│   ├── anyattack_lite.py    # Learned perturbation generator
│   └── text_targeted.py
├── train/
│   ├── train_anyattack.py   # Generator training loop
│   └── train.py             # Helper training entry
├── models/
│   └── clip_wrapper.py
├── eval/
│   └── metrics.py           # ASR computation
├── utils/
│   ├── loss.py
│   ├── image.py
│   └── seed.py
├── data/
│   └── dogs/                # Example training / test images
├── requirements.txt
└── README.md

```

---

## Setup

### 1. Create environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

If CLIP fails to install:

```bash
pip install git+https://github.com/openai/CLIP.git
```

---

## Running Attacks (No Training Required)

### Text-Targeted PGD

```bash
python main.py
```

Outputs:

* `adv_pgd.png`
* printed original vs target probabilities
* success flag

This already satisfies the **baseline attack requirement** of the assignment.

---

## Training AnyAttack-lite (Required Before Testing)

Model weights are **not included**.
You must train the generator locally.

### 1. Prepare data

Place images in:

```
data/dogs/
```

Small datasets (5–20 images) are acceptable for assignment-scale experiments.

---

### 2. Train the generator

```bash
python train/train_anyattack.py
```

What happens:

* CLIP is frozen
* Generator learns adversarial perturbations via CLIP similarity loss
* No labels or annotations required

After training, a checkpoint is saved locally (e.g. `anyattack.pt`).

---

### 3. Run inference using trained weights

Update `main.py` to load the trained generator, then run:

```bash
python main.py
```

Outputs:

* `adv_anyattack.png`
* attack success metrics

---

## Expected Results (Important)

* **PGD** usually outperforms AnyAttack-lite on small datasets
* **AnyAttack-lite may fail** when trained on very few images
* This is expected and discussed in the report as a limitation of universal / weakly-conditioned attacks

Failure cases are **valid experimental findings**, not bugs.

---

## Metrics

Attack performance is measured using:

* **Target probability**
* **Attack Success Rate (ASR)**

Defined in:

```
eval/metrics.py
```

Success criterion:

```
P(target text) > P(original text)
```

---

## Notes on Reproducibility

* No pretrained adversarial weights are provided
* All attacks are fully reproducible from code
* CLIP weights are automatically downloaded
* Docker is intentionally omitted (per assignment constraints)

---

## Academic Context

This repo is inspired by:

* *AnyAttack: Towards Universal Adversarial Attacks*
* Prior work on CLIP robustness and VLM attacks

This implementation is **not a full reproduction** of AnyAttack, but a **faithful research prototype** adapted for academic evaluation.

---

## License

MIT License
