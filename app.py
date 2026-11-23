import os, sys, time, re, glob, json, numpy as np
from typing import Optional, List
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import sign_model, llm

app = FastAPI(title="SignBridge")
rec = sign_model.Recognizer()
sessions = {}  # (session id, mode) -> history


class Req(BaseModel):
    session: str = "s"
    mode: str = "ours"
    frames: Optional[list] = None
    simulate: Optional[dict] = None
    label: Optional[str] = None
    signer: Optional[str] = None


class HearingReq(BaseModel):
    session: str = "s"
    mode: str = "ours"
    message: Optional[str] = None
    text: Optional[str] = None


class FeedbackReq(BaseModel):
    session: str = "s"
    turn_id: Optional[str] = None
    rating: int = 5  # 1 to 5
    aspect: Optional[str] = "overall"
    metric: Optional[str] = None
    comment: Optional[str] = None
    notes: Optional[str] = None


def _history(r: Req):
    return sessions.setdefault((r.session, r.mode), [])


def _parse_simulated(sim: dict):
    if sim.get("segments"):
        segs = []
        for s in sim["segments"]:
            g = str(s.get("gloss", "")).strip().upper()
            if g:
                segs.append({"gloss": g, "conf": float(s.get("conf", 0.9)),
                             "alts": [[str(a[0]).upper(), float(a[1])] for a in s.get("alts", []) if a]})
        return segs
    glosses = [str(x).strip().upper() for x in sim.get("glosses", []) if str(x).strip()]
    c = float(sim.get("conf", 0.9))
    alts = []
    for a in sim.get("alts", []) or []:
        if isinstance(a, (list, tuple)) and a:
            alts.append([str(a[0]).upper(), float(a[1]) if len(a) > 1 else round(1 - c, 2)])
        elif str(a).strip():
            alts.append([str(a).strip().upper(), round(1 - c, 2)])
    last = len(glosses) - 1
    return [{"gloss": g, "conf": c, "alts": alts} if i == last else {"gloss": g, "conf": max(c, 0.95), "alts": []}
            for i, g in enumerate(glosses)]


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/status")
def status():
    global rec
    if not rec.ready and os.path.exists(sign_model.MODEL_PATH):
        rec = sign_model.Recognizer()
    b = llm.get_backend()
    n_rec = len(glob.glob(os.path.join(BASE_DIR, "data", "recorded", "*", "*.npy")))
    n_h2s = len(glob.glob(os.path.join(BASE_DIR, "data", "how2sign", "*", "*.npy")))
    return {
        "model_ready": rec.ready,
        "vocab": rec.classes,
        "vocab_count": len(rec.classes),
        "trained_on": rec.data_source,
        "temperature": rec.temperature,
        "kinematic_priors": sign_model.USE_PRIORS,
        "llm_backend": b.name,
        "llm_is_real": b.is_llm,
        "llm_approach": getattr(llm, "MEMBER_1_APPROACH", {}),
        "policy": {"low": llm.LOW, "high": llm.HIGH, "min_margin": llm.MIN_MARGIN},
        "datasets": {
            "recorded_count": n_rec,
            "how2sign_count": n_h2s,
            "total_count": n_rec + n_h2s,
            "integrated": ["How2Sign OpenPose 2D Keypoints (val, test, train)", "Browser MediaPipe WebCam Recordings"]
        }
    }


@app.get("/api/competition")
def competition():
    return {
        "active_member": "Member 1",
        "active_model": "Qwen2.5-3B-Instruct",
        "active_technique": "Plain Prompting (Zero-Shot Baseline)",
        "approaches": getattr(llm, "TEAM_APPROACHES", {}),
        "verdict_criteria": [
            "Task Success Rate on clear and ambiguous turns",
            "Hallucination / Wrong Answer Rate (Hypothesis 2)",
            "Interpretation Accuracy (Hypothesis 1)",
            "Edge Latency and Parameter Footprint (running locally on laptops)"
        ]
    }


@app.post("/api/hearing_turn")
def hearing_turn(r: HearingReq):
    hist = sessions.setdefault((r.session, r.mode), [])
    msg = (r.text or r.message or "").strip()
    if not msg:
        return {"error": "Empty message"}
    hist.append({"role": "hearing_user", "content": msg, "timestamp": time.time()})
    return {"ok": True, "status": "ok", "speaker": "hearing_interlocutor", "message": msg}


@app.post("/api/feedback")
def feedback(r: FeedbackReq):
    fb_path = os.path.join(BASE_DIR, "eval", "feedback.jsonl")
    os.makedirs(os.path.dirname(fb_path), exist_ok=True)
    with open(fb_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "session": r.session,
            "turn_id": r.turn_id,
            "rating": r.rating,
            "metric": r.metric or r.aspect,
            "comment": r.comment or r.notes,
            "timestamp": time.time()
        }) + "\n")
    return {"saved": True, "rating": r.rating}


@app.post("/api/reset")
def reset(r: Req):
    for key in [k for k in sessions if k[0] == r.session]:
        del sessions[key]
    return {"ok": True}


@app.post("/api/collect")
def collect(r: Req):
    if not r.label or not r.label.strip():
        return {"saved": False, "msg": "Please enter a sign label (e.g. LIBRARY)"}
    signer = re.sub(r"[^a-z0-9]", "", (r.signer or "").lower())
    if not signer:
        return {"saved": False, "msg": "Please enter a signer id (e.g. your first name) so we can test on unseen people"}
    if not r.frames or len(r.frames) < 8:
        return {"saved": False, "msg": "Recording too short or no frames received"}
    x = np.asarray(r.frames, dtype=np.float32)
    p = sign_model.hands_present(x)
    if p.sum() < 8:
        return {"saved": False, "msg": "No hands detected in recording"}
    x = x[np.where(p)[0][0]: np.where(p)[0][-1] + 1]
    label = re.sub(r"[^A-Z0-9-]", "", r.label.strip().upper())
    d = os.path.join(BASE_DIR, "data", "recorded", label)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"{signer}__{int(time.time() * 1000)}.npy"), x)
    n = len([f for f in os.listdir(d) if f.endswith(".npy")])
    mine = len([f for f in os.listdir(d) if f.startswith(signer + "__")])
    return {"saved": True, "count": n, "count_signer": mine}


@app.post("/api/interpret")
def interpret(r: Req):
    global rec
    if r.mode not in llm.MODES:
        return {"error": f"Unknown mode '{r.mode}'"}
    if r.simulate:
        segs = _parse_simulated(r.simulate)
        if not segs:
            return {"error": "No glosses given"}
        cues = {"eyebrows_raised": bool(r.simulate.get("eyebrows_raised", False)), "mouth_open": False}
    else:
        if not rec.ready and os.path.exists(sign_model.MODEL_PATH):
            rec = sign_model.Recognizer()
        if not r.frames or len(r.frames) < 8:
            return {"error": "Recording too short: keep your hands in frame and sign clearly."}
        if not rec.ready:
            return {"error": "No trained model yet: record signs in the Collect panel, then run `python train.py`."}
        segs, cues = rec.predict(r.frames)
        if not segs:
            return {"error": "No signs detected. Keep your hands visible and pause briefly between signs."}

    hist = _history(r)
    res = llm.ask(hist, segs, cues, r.mode)
    llm.record_turn(hist, segs, res)
    return {"segments": segs, "confidence": res["confidence"], "cues": cues, "tier": res["tier"], "result": res}


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
