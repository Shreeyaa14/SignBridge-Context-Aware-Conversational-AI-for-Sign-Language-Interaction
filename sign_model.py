"""Sign recognizer: landmark sequence -> segments -> top-k glosses + confidence (BiGRU).

Confidence = softmax(logits / T). T is fitted on a validation split by train.py
(temperature scaling, Guo et al. 2017) and stored in the checkpoint; old
checkpoints without it fall back to T = 1.0 (uncalibrated).

The hand-written "kinematic priors" (compute_kinematic_priors) are OFF by default.
They add +2..+7 to the logits based on rules tuned on synthetic data, which
overrides the network and makes the confidence meaningless. Turn them on only
for experiments: SIGNBRIDGE_KINEMATIC_PRIORS=1
"""
import os, numpy as np, torch, torch.nn as nn

F, L = 149, 32            # features per frame, resampled length
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "sign_model.pt")
USE_PRIORS = os.getenv("SIGNBRIDGE_KINEMATIC_PRIORS", "0") == "1"
TOP_K = 3
DEFAULT_VOCAB = ["HELLO","GOODBYE","YES","NO","HELP","THANK-YOU","SORRY","UNDERSTAND",
    "NOT-UNDERSTAND","REPEAT","WHERE","WHAT","WHO","LIBRARY","CLASSROOM","BATHROOM",
    "FOOD","WATER","DOCTOR","APPOINTMENT","TOMORROW","OFFICE","WHEN"]

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)

class SignGRU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.gru = nn.GRU(F, 128, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = TemporalAttention(256)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.3), nn.Linear(256, n))
    def forward(self, x):
        o, _ = self.gru(x)
        pooled = 0.6 * self.attn(o) + 0.4 * o.mean(1)
        return self.head(pooled)

def resample(x, n=L):
    if len(x) == 0:
        return np.zeros((n, x.shape[1] if x.ndim > 1 else F), dtype=np.float32)
    if len(x) == 1:
        return np.repeat(x, n, axis=0).astype(np.float32)
    t0, t1 = np.linspace(0, 1, len(x)), np.linspace(0, 1, n)
    return np.stack([np.interp(t1, t0, x[:, i]) for i in range(x.shape[1])], 1).astype(np.float32)

def hands_present(x):
    """
    Detect hands actively engaged in the ASL signing space (chest to head level Y > -0.72).
    Ignores hands resting on the desk, lap, or trackpad (Y <= -0.75).
    """
    has_lh = (x[:, 1] > -0.72) & (np.abs(x[:, :63]).sum(1) > 0)
    has_rh = (x[:, 64] > -0.72) & (np.abs(x[:, 63:126]).sum(1) > 0)
    p = has_lh | has_rh
    if not p.any():
        # Fallback for lower torso gestures if no chest-height landmarks
        has_lh = (x[:, 1] > -0.85) & (np.abs(x[:, :63]).sum(1) > 0)
        has_rh = (x[:, 64] > -0.85) & (np.abs(x[:, 63:126]).sum(1) > 0)
        p = has_lh | has_rh
    if not p.any():
        p = np.abs(x[:, :126]).sum(1) > 0
    return p

def trim_sign_boundary(seg, y_thresh=-0.72):
    """
    Trim non-signing boundary frames (e.g. mouse clicks or hands moving up/down from desk)
    to focus strictly on the active ASL gesture.
    """
    lh_in = (seg[:, 1] > y_thresh) & (np.abs(seg[:, :63]).sum(1) > 0)
    rh_in = (seg[:, 64] > y_thresh) & (np.abs(seg[:, 63:126]).sum(1) > 0)
    act = lh_in | rh_in
    if act.sum() >= 6:
        idx = np.where(act)[0]
        return seg[idx[0]:idx[-1] + 1]
    return seg

def segments(x, gap=8, min_len=8):
    """Split a recording into signs: contiguous hand-visible runs separated by >=gap empty frames.
    Merges jitter-split segments closer than gap frames to avoid over-segmentation.
    """
    p, raw, start, miss = hands_present(x), [], None, 0
    for i, v in enumerate(p):
        if v:
            if start is None: start = i
            miss = 0
        elif start is not None:
            miss += 1
            if miss >= gap:
                if i - miss + 1 - start >= min_len:
                    raw.append((start, i - miss + 1))
                start, miss = None, 0
    if start is not None and len(p) - start - miss >= min_len:
        raw.append((start, len(p) - miss))

    # Resilient fallback: if no gap-separated segment reached min_len but hands are clearly present
    if not raw and p.sum() >= min_len:
        idx = np.where(p)[0]
        raw.append((int(idx[0]), int(idx[-1] + 1)))

    if not raw:
        return []

    # Merge consecutive clusters separated by fewer than gap frames
    merged = [raw[0]]
    for cur in raw[1:]:
        prev = merged[-1]
        if cur[0] - prev[1] < gap:
            merged[-1] = (prev[0], cur[1])
        else:
            merged.append(cur)

    return merged

def compute_kinematic_priors(seg, classes):
    """
    Computes invariant ASL physical kinematic prior bonuses based on:
    movement direction (dy, dx), handshape (fist vs open vs index vs pinch),
    signing space location (face, chin, chest, torso), and one/two-handedness.
    """
    scores = np.zeros(len(classes), dtype=np.float32)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    if len(seg) < 4:
        return scores

    rh = seg[:, 63:126].reshape(len(seg), 21, 3)
    lh = seg[:, 0:63].reshape(len(seg), 21, 3)

    rh_act = np.abs(rh).sum(axis=(1, 2)) > 0
    lh_act = np.abs(lh).sum(axis=(1, 2)) > 0

    if not rh_act.any():
        if lh_act.any():
            rh = lh
            rh_act = lh_act
            lh_act = np.zeros(len(seg), dtype=bool)
        else:
            return scores

    rh_frames = rh[rh_act]
    w = rh_frames[:, 0]
    n_pts = len(w)
    head_k = max(1, n_pts // 4)
    tail_k = max(1, n_pts // 4)
    y_start = float(w[:head_k, 1].mean())
    y_end = float(w[-tail_k:, 1].mean())
    dy = y_end - y_start
    dx = float(w[-tail_k:, 0].mean()) - float(w[:head_k, 0].mean())

    mean_y = float(w[:, 1].mean())
    mean_x = float(w[:, 0].mean())
    var_x = float(w[:, 0].var())
    var_y = float(w[:, 1].var())

    tips = [4, 8, 12, 16, 20]
    tip_dists = np.linalg.norm(rh_frames[:, tips] - rh_frames[:, 0:1], axis=-1).mean(axis=0)
    avg_finger_ext = float(tip_dists[1:].mean())
    index_ext = float(tip_dists[1])
    other_ext = float(tip_dists[2:].mean())

    is_fist = bool(avg_finger_ext < 0.26)
    is_open = bool(avg_finger_ext >= 0.26)
    is_index = bool(index_ext > 0.26 and other_ext < 0.22)
    two_handed = bool(lh_act.sum() >= 4 and rh_act.sum() >= 4)

    def bonus(gloss, val):
        if gloss in cls_to_idx:
            scores[cls_to_idx[gloss]] += val

    # 1. HELP: Dominant fist lifting UPWARDS (dy > 0.02) in chest/torso area
    if dy > 0.02 and (is_fist or not is_open or dy > 0.06):
        bonus("HELP", 3.0 + 4.0 * min(1.0, dy / 0.12))
        if two_handed:
            bonus("HELP", 2.0)

    # 2. THANK-YOU: Open hand moving DOWNWARD / forward from chin/lips (dy < -0.02)
    if dy < -0.02 and mean_y > -0.15 and (is_open or not is_fist):
        bonus("THANK-YOU", 3.0 + 4.0 * min(1.0, -dy / 0.12))

    # 3. SORRY: Fist circular motion on chest
    if is_fist and abs(dy) < 0.06 and var_x > 0.0003 and var_y > 0.0003 and mean_y < 0.10:
        bonus("SORRY", 3.5)

    # 4. YES: Fist vertical nodding motion
    if is_fist and var_y > 0.0004 and abs(dx) < 0.05 and abs(dy) < 0.06:
        bonus("YES", 3.0)

    # 5. NO: Snapping/pinching fingers at chest
    if abs(dy) < 0.06 and not is_open:
        bonus("NO", 2.0)

    # 6. WATER: At chin/lips (mean_y > 0.05), fingers extended, small net displacement
    if mean_y > 0.05 and abs(mean_x) < 0.25 and abs(dy) < 0.05 and not is_fist:
        bonus("WATER", 2.5)

    # 7. FOOD: At mouth/lips, pinched fingertips tapping mouth
    if mean_y > 0.08 and abs(mean_x) < 0.20 and abs(dy) < 0.05 and not is_open:
        bonus("FOOD", 2.5)

    # 8. HELLO: Temple/forehead level, open hand waving/saluting outward
    if mean_y > 0.22 and mean_x > 0.12 and is_open and dy >= -0.03:
        bonus("HELLO", 3.5)

    # 9. GOODBYE: Head/shoulder level, open hand waving side to side
    if mean_y > 0.15 and is_open and var_x > 0.0005:
        bonus("GOODBYE", 3.0)

    # 10. WHAT: Two-handed, open hands at chest/torso
    if two_handed and mean_y < 0.10 and is_open:
        bonus("WHAT", 3.5)

    # 11. WHERE: Index finger pointing up, shaking side to side
    if is_index and var_x > 0.0003 and mean_y > -0.15:
        bonus("WHERE", 3.5)

    # 12. BATHROOM: Fist shaking side-to-side at chest/shoulder level
    if is_fist and var_x > 0.0004 and abs(dy) < 0.05:
        bonus("BATHROOM", 3.0)

    # 13. UNDERSTAND: Temple level, index extended flicking up
    if mean_y > 0.18 and is_index and dy > 0.02:
        bonus("UNDERSTAND", 3.5)

    return scores


class Recognizer:
    def __init__(self):
        self.model, self.classes = None, DEFAULT_VOCAB
        self.temperature, self.data_source = 1.0, None
        self._mtime = 0
        self.load()

    def load(self):
        if os.path.exists(MODEL_PATH):
            self._mtime = os.path.getmtime(MODEL_PATH)
            ck = torch.load(MODEL_PATH, map_location="cpu")
            self.classes = ck["classes"]
            self.temperature = float(ck.get("temperature", 1.0))
            # checkpoints from before this change were trained on the synthetic generator
            self.data_source = ck.get("data_source", "synthetic (legacy checkpoint)")
            self.model = SignGRU(len(self.classes))
            self.model.load_state_dict(ck["state"])
            self.model.eval()

    def check_reload(self):
        if os.path.exists(MODEL_PATH):
            m = os.path.getmtime(MODEL_PATH)
            if m > getattr(self, "_mtime", 0):
                self.load()

    @property
    def ready(self):
        self.check_reload()
        return self.model is not None

    def predict(self, frames):
        self.check_reload()
        if not self.ready:
            return [], {}
        x = np.asarray(frames, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != F or len(x) < 8:
            return [], {}

        out = []
        brow = x[:, 147][x[:, 147] > 0]
        mouth = x[:, 148]
        base = float(np.median(brow)) if len(brow) else 0.0

        for a, b in segments(x):
            seg = trim_sign_boundary(x[a:b])
            if len(seg) < 2:
                continue

            # Candidate 1: Raw trimmed segment
            candidates = [seg.copy()]

            # Determine dominant hand motion and vertical positions
            lh_act = np.abs(seg[:, :63]).sum(axis=1) > 0
            rh_act = np.abs(seg[:, 63:126]).sum(axis=1) > 0
            lh_motion = float(np.var(seg[lh_act, :63])) if lh_act.sum() > 2 else 0.0
            rh_motion = float(np.var(seg[rh_act, 63:126])) if rh_act.sum() > 2 else 0.0
            lh_y = float(seg[lh_act, 1].mean()) if lh_act.sum() > 2 else -1.0
            rh_y = float(seg[rh_act, 64].mean()) if rh_act.sum() > 2 else -1.0

            # Candidate 2: Resting hand suppression (ignores inactive hand on desk/trackpad)
            if rh_motion > 0.0003 and (lh_motion < 0.25 * rh_motion or lh_y < -0.68):
                c_rh = seg.copy()
                c_rh[:, :63] = 0.0
                candidates.append(c_rh)
            if lh_motion > 0.0003 and (rh_motion < 0.25 * lh_motion or rh_y < -0.68):
                c_lh = seg.copy()
                c_lh[:, 63:126] = 0.0
                candidates.append(c_lh)

            # Candidates 3 & 4: Ambidextrous mirror mappings (left-handed signers & camera mirroring)
            for c in list(candidates):
                cm = c.copy()
                cm[:, :63] = c[:, 63:126]
                cm[:, 63:126] = c[:, :63]
                cm[:, 0:126:3] = -cm[:, 0:126:3]
                candidates.append(cm)

            best_logits = None
            best_max = -1e9
            with torch.no_grad():
                for cand in candidates:
                    inp = torch.from_numpy(resample(cand))[None]
                    logits = self.model(inp)[0].numpy()
                    if USE_PRIORS:
                        logits = logits + compute_kinematic_priors(cand, self.classes)
                    m = float(logits.max())
                    if m > best_max:
                        best_max = m
                        best_logits = logits

            pr = torch.softmax(torch.from_numpy(best_logits) / self.temperature, -1).numpy()
            idx = pr.argsort()[::-1][:TOP_K]
            out.append({
                "gloss": self.classes[idx[0]],
                "conf": float(pr[idx[0]]),
                "alts": [[self.classes[i], float(pr[i])] for i in idx[1:]]
            })

        b = x[:, 147]
        cues = {
            "eyebrows_raised": bool(len(brow) > 0 and base > 0 and float(b.max()) > 1.12 * base),
            "mouth_open": bool(float(mouth.max()) > 0.08) if len(mouth) else False
        }
        return out, cues

