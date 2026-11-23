"""Train the BiGRU sign recognizer.

    python train.py                          # recorded data (data/recorded/<GLOSS>/*.npy)
    python train.py --source synthetic       # generator output only (pipeline smoke test, NOT a result)
    python train.py --holdout-signer anna    # test on one person the model never saw

Each .npy is a (T, 149) landmark sequence from the web app's "Collect" panel.
Recorded files are named <signer>__<timestamp>.npy so we can split by person.

Steps: train on the train split, fit a softmax temperature on the validation
split (temperature scaling, Guo et al. 2017) so that confidence values mean
something, then report accuracy / top-3 / ECE on the test split.
"""
import argparse, glob, json, os, random, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sign_model import SignGRU, resample, MODEL_PATH

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["recorded", "how2sign", "real", "synthetic", "all"], default="real")
ap.add_argument("--holdout-signer", default=None, help="signer id used only for testing")
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
roots = {
    "recorded": ["data/recorded"],
    "how2sign": ["data/how2sign"],
    "real": ["data/recorded", "data/how2sign"],
    "synthetic": ["data/synthetic"],
    "all": ["data/recorded", "data/how2sign", "data/synthetic"],
}[args.source]

files = []  # (path, gloss, signer)
for root in roots:
    for f in glob.glob(f"{root}/*/*.npy"):
        gloss = os.path.basename(os.path.dirname(f))
        if gloss in ("THANK-YOU", "THANKYOU"):
            gloss = "THANKYOU"
        elif gloss in ("NOT-UNDERSTAND", "NOTUNDERSTAND"):
            gloss = "NOT-UNDERSTAND"
        name = os.path.basename(f)
        signer = name.split("__")[0] if "__" in name else ("synthetic" if "synthetic" in root else "unknown")
        files.append((f, gloss, signer))

if not files:
    raise SystemExit(f"No samples found in {roots}. Record signs with the web app's Collect panel "
                     f"(saved to data/recorded/<GLOSS>/), or run with --source synthetic for a smoke test.")

classes = sorted({g for _, g, _ in files})
cidx = {c: i for i, c in enumerate(classes)}
X = [np.load(f).astype(np.float32) for f, _, _ in files]
y = np.array([cidx[g] for _, g, _ in files])
signers = np.array([s for _, _, s in files])
print(f"source={args.source}: {len(classes)} classes, {len(X)} samples, signers={sorted(set(signers))}")
if args.source == "synthetic" or "data/synthetic" in roots:
    print("WARNING: synthetic samples come from scripts/prepare_dataset.py, not from people signing. "
          "Accuracy on them says nothing about real webcam input.")

idx = np.arange(len(X))
def strat(labels):
    return labels if np.bincount(labels).min() >= 2 else None

if args.holdout_signer:
    te = idx[signers == args.holdout_signer]
    rest = idx[signers != args.holdout_signer]
    if len(te) == 0:
        raise SystemExit(f"No samples from signer '{args.holdout_signer}'.")
    tr, va = train_test_split(rest, test_size=0.15, stratify=strat(y[rest]), random_state=args.seed)
else:
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=strat(y), random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=strat(y[tmp]), random_state=args.seed)
print(f"split: train={len(tr)} val={len(va)} test={len(te)}")


def aug(x):
    n = len(x)
    a = random.randint(0, max(0, n // 8))
    b = n - random.randint(0, max(0, n // 8))
    x = resample(x[a:max(b, a + 4)]).copy()

    # 1. Ambidextrous mirror augmentation (50% chance)
    if random.random() < 0.5:
        xm = x.copy()
        xm[:, :63] = x[:, 63:126]
        xm[:, 63:126] = x[:, :63]
        xm[:, 0:126:3] = -xm[:, 0:126:3]
        x = xm

    # 2. Resting hand noise augmentation (35% chance)
    # If one hand is inactive, simulate resting hand at desk/trackpad level
    lh_zero = (np.abs(x[:, :63]).sum() == 0)
    rh_zero = (np.abs(x[:, 63:126]).sum() == 0)
    if random.random() < 0.35:
        if lh_zero and not rh_zero:
            x[:, :63] = np.random.normal(0, 0.02, (len(x), 63))
            x[:, 1:63:3] = np.random.uniform(-0.85, -0.72)
        elif rh_zero and not lh_zero:
            x[:, 63:126] = np.random.normal(0, 0.02, (len(x), 63))
            x[:, 64:126:3] = np.random.uniform(-0.85, -0.72)

    # 3. Geometric scaling, jitter, translation
    present = (x[:, :147] != 0)            # keep missing hands at exactly 0, as at inference time
    x[:, :147] *= np.random.uniform(0.88, 1.12)
    shift_x, shift_y = np.random.uniform(-0.05, 0.05), np.random.uniform(-0.04, 0.04)
    x[:, 0:147:3] += shift_x
    x[:, 1:147:3] += shift_y
    x = x + np.random.normal(0, 0.012, x.shape).astype(np.float32)
    x[:, :147] *= present
    return x


device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
m = SignGRU(len(classes)).to(device)
opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

for ep in range(args.epochs):
    m.train()
    order = list(tr)
    random.shuffle(order)
    for i in range(0, len(order), 16):
        b = order[i:i + 16]
        xb = torch.tensor(np.stack([aug(X[j]) for j in b]), dtype=torch.float32, device=device)
        yb = torch.tensor(y[b], dtype=torch.long, device=device)
        opt.zero_grad()
        lossf(m(xb), yb).backward()
        opt.step()
    sched.step()


@torch.no_grad()
def logits_for(ids):
    m.eval()
    xb = torch.tensor(np.stack([resample(X[j]) for j in ids]), dtype=torch.float32, device=device)
    return m(xb).cpu()


def fit_temperature(logits, labels):
    """Temperature scaling: one scalar T minimising validation NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    lab = torch.tensor(labels, dtype=torch.long)
    o = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    def closure():
        o.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), lab)
        loss.backward()
        return loss
    o.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))


def ece(probs, labels, bins=10):
    """Expected calibration error: gap between confidence and accuracy, averaged over bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


T = fit_temperature(logits_for(va), y[va])
lo_te = logits_for(te)
p_raw = torch.softmax(lo_te, -1).numpy()
p_cal = torch.softmax(lo_te / T, -1).numpy()
pred = p_cal.argmax(1)
k = min(3, len(classes))
top3 = float(np.mean([t in r for t, r in zip(y[te], lo_te.topk(k).indices.numpy())]))
metrics = {
    "source": args.source,
    "holdout_signer": args.holdout_signer,
    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
    "test_accuracy": float((pred == y[te]).mean()),
    "test_top3_accuracy": top3,
    "temperature": T,
    "ece_before_calibration": ece(p_raw, y[te]),
    "ece_after_calibration": ece(p_cal, y[te]),
}
print(classification_report(y[te], pred, labels=list(range(len(classes))),
                            target_names=classes, zero_division=0))
print(json.dumps(metrics, indent=2))

os.makedirs("model", exist_ok=True)
torch.save({"state": m.cpu().state_dict(), "classes": classes, "temperature": T,
            "data_source": args.source, "metrics": metrics}, MODEL_PATH)
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Model saved to {MODEL_PATH} (metrics in model/metrics.json)")
