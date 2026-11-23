"""Language layer for SignBridge: recognized signs -> interpretation + reply.

Three systems are compared (see README, "Experimental comparison"):

  Model A  "direct"  Glosses joined into text. No LLM, no context, never asks.
  Model B  "llm"     LLM gets the top-1 glosses + conversation history.
                     It is told nothing about recognition confidence.
  Model C  "ours"    LLM gets top-k candidates with confidence + history, and a
                     confidence policy is enforced in code:
                        min confidence <  LOW  or top-2 margin < MIN_MARGIN -> ask (no LLM call)
                        LOW <= confidence < HIGH                           -> LLM decides, may ask
                        confidence >= HIGH                                 -> LLM responds
  Ablation "ours_nogate": same prompt as C but without the code gate, to test
                     whether the LLM uses confidence well on its own.

B and C use the same knowledge base, output schema and backend, so any
difference between them comes from the confidence information and the gate.

Backends (env SIGNBRIDGE_LLM = auto | ollama | openai | anthropic | rules):
  ollama     local Ollama, model OLLAMA_MODEL (default qwen2.5:3b-instruct)
  openai     any OpenAI-compatible API: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
  anthropic  ANTHROPIC_API_KEY, ANTHROPIC_MODEL
  rules      deterministic keyword rules, NOT an LLM. Used only when no LLM is
             available, and always labelled as such. Do not report rules output
             as Model B/C results.
"""
import json, os, re, time
import requests

# --------------------------------------------------------------------------- policy
LOW = float(os.getenv("SIGNBRIDGE_LOW", "0.50"))
HIGH = float(os.getenv("SIGNBRIDGE_HIGH", "0.80"))
MIN_MARGIN = float(os.getenv("SIGNBRIDGE_MIN_MARGIN", "0.10"))
HISTORY_TURNS = 6
MODES = ("direct", "llm", "ours", "ours_nogate")

# --------------------------------------------------------------------------- domain
ACTS = ["ask_location", "ask_time", "ask_who", "ask_info", "request_help", "greet", "farewell",
        "thank", "apologize", "affirm", "deny", "request_repeat", "not_understood", "understood",
        "unknown"]
TOPICS = ["LIBRARY", "CLASSROOM", "BATHROOM", "OFFICE", "DOCTOR", "APPOINTMENT", "FOOD", "WATER", "NONE"]
QUESTION_WORDS = {"WHERE": "ask_location", "WHEN": "ask_time", "WHO": "ask_who", "WHAT": "ask_info"}

KB = {
    "LIBRARY": {"name": "library", "location": "Building B, second floor",
                "hours": "Monday-Friday 9:00-18:00; study cubicles 24/7 with a student badge",
                "who": "the library desk staff can help with books, printing and study rooms"},
    "CLASSROOM": {"name": "classrooms", "location": "Building A, first floor (rooms A101-A125)",
                  "hours": "lectures run weekdays 8:30-17:00",
                  "who": "your course coordinator or the student office can tell you your room"},
    "BATHROOM": {"name": "bathroom", "location": "next to the main entrance on the ground floor, and next to the stairwells on every floor",
                 "hours": "open whenever the building is open",
                 "who": "facility staff at the main entrance"},
    "OFFICE": {"name": "student office", "location": "Building A, second floor, room A204",
               "hours": "Monday-Friday 9:00-17:00",
               "who": "student administration: ID cards, enrolment certificates and transcripts"},
    "DOCTOR": {"name": "campus doctor", "location": "Building C, room 12 (Student Health Centre)",
               "hours": "walk-in 9:00-12:00, appointments 13:00-16:30",
               "who": "the campus GP and a nurse"},
    "APPOINTMENT": {"name": "appointment", "location": "Student Health Centre, Building C, room 12",
                    "hours": "tomorrow at 10:00 (please arrive 10 minutes early with your student ID)",
                    "who": "the campus GP"},
    "FOOD": {"name": "cafeteria", "location": "Building B, ground floor",
             "hours": "daily 8:00-19:00",
             "who": "the cafeteria serves hot meals, sandwiches and vegetarian options"},
    "WATER": {"name": "water fountains", "location": "on every floor, next to the lifts and bathrooms",
              "hours": "available 24/7",
              "who": "facility staff if a fountain is broken"},
}
WORD = {k: v["name"] for k, v in KB.items()}
WORD.update({"HELLO": "hello", "GOODBYE": "goodbye", "YES": "yes", "NO": "no", "HELP": "help",
             "THANK-YOU": "thank you", "SORRY": "sorry", "UNDERSTAND": "understand",
             "NOT-UNDERSTAND": "don't understand", "REPEAT": "repeat", "WHERE": "where",
             "WHAT": "what", "WHO": "who", "WHEN": "when", "TOMORROW": "tomorrow"})


def tier(c: float) -> str:
    return "high" if c >= HIGH else "medium" if c >= LOW else "low"


def _alts(seg):
    out = []
    for a in seg.get("alts") or []:
        if isinstance(a, (list, tuple)) and a:
            out.append((str(a[0]).upper(), float(a[1]) if len(a) > 1 else 0.0))
        elif isinstance(a, str):
            out.append((a.upper(), 0.0))
    return out


def assess(segs):
    """Apply the confidence policy. Returns the decision and the least certain sign."""
    if not segs:
        return {"decision": "clarify", "confidence": 0.0, "margin": 0.0, "weakest": None, "tier": "low"}
    rows = []
    for i, s in enumerate(segs):
        alts = _alts(s)
        margin = s["conf"] - (alts[0][1] if alts else 0.0)
        rows.append((s["conf"], margin, i))
    conf = min(r[0] for r in rows)
    margin = min(r[1] for r in rows)
    weakest = min(rows, key=lambda r: (r[0], r[1]))[2]
    if conf < LOW or margin < MIN_MARGIN:
        decision = "clarify"
    elif conf < HIGH:
        decision = "llm_decides"
    else:
        decision = "respond"
    return {"decision": decision, "confidence": conf, "margin": margin, "weakest": weakest, "tier": tier(conf)}


def clarification_question(segs, weakest):
    if not segs or weakest is None:
        return "I couldn't see any signs clearly. Could you sign again with your hands in the frame?"
    s = segs[weakest]
    cands = [s["gloss"]] + [g for g, p in _alts(s) if p >= 0.05 and g != s["gloss"]]
    words = [WORD.get(g, g.lower()) for g in cands[:3]]
    if len(words) == 1:
        return f"I'm not sure I recognised \"{words[0]}\" correctly. Could you sign it again?"
    options = ", ".join(f"\"{w}\"" for w in words[:-1]) + f" or \"{words[-1]}\""
    return f"I'm not sure whether you signed {options}. Which one did you mean?"


# --------------------------------------------------------------------------- history helpers
def history_topic(history):
    """Most recent topic the student talked about (from stored intents)."""
    for m in reversed(history[-2 * HISTORY_TURNS:]):
        t = (m.get("intent") or {}).get("topic")
        if m.get("role") == "user" and t and t != "NONE":
            return t
    return None


def pending_clarification(history):
    """If the assistant's last turn was a clarification question, return what it was about."""
    if history and history[-1].get("role") == "assistant" and history[-1].get("clarification"):
        return history[-1]
    return None


def transcript(history):
    lines = []
    for m in history[-2 * HISTORY_TURNS:]:
        role = m.get("role")
        if role == "user":
            who = "Student (signed)"
            lines.append(f"{who}: {m.get('content', '')}")
        elif role == "hearing_user":
            who = "Hearing staff (spoken/typed)"
            lines.append(f"{who}: \"{m.get('content', '')}\"")
        else:
            who = "Assistant"
            lines.append(f"{who}: {m.get('content', '')}")
    return "\n".join(lines) if lines else "(no earlier turns)"


# --------------------------------------------------------------------------- literal / rule-based reading
def parse_intent(glosses, ctx_topic=None, pending=None):
    """Keyword reading of a gloss sequence. Used for Model A scoring and the rules fallback."""
    g = [x.upper() for x in glosses]
    topic = "APPOINTMENT" if "APPOINTMENT" in g or ("DOCTOR" in g and "TOMORROW" in g) else \
        next((x for x in g if x in KB), None)
    qword = next((QUESTION_WORDS[x] for x in g if x in QUESTION_WORDS), None)

    if pending is not None and topic and not qword:
        # answer to "did you mean X or Y?" -> keep the act of the pending question
        prev = pending.get("pending_intent") or {}
        return {"act": prev.get("act") if prev.get("act") not in (None, "unknown") else "ask_location",
                "topic": topic}
    if qword:
        return {"act": qword, "topic": topic or ctx_topic or "NONE"}
    for gloss, act in (("REPEAT", "request_repeat"), ("NOT-UNDERSTAND", "not_understood"),
                       ("NOTUNDERSTAND", "not_understood")):
        if gloss in g:
            return {"act": act, "topic": "NONE"}
    if topic:
        return {"act": "ask_location", "topic": topic}
    for gloss, act in (("HELP", "request_help"), ("THANK-YOU", "thank"), ("THANKYOU", "thank"),
                       ("GOODBYE", "farewell"), ("HELLO", "greet"), ("SORRY", "apologize"),
                       ("UNDERSTAND", "understood"), ("YES", "affirm"), ("NO", "deny")):
        if gloss in g:
            return {"act": act, "topic": "NONE"}
    return {"act": "unknown", "topic": "NONE"}


def intent_to_text(intent):
    act, t = intent["act"], intent["topic"]
    name = KB[t]["name"] if t in KB else ""
    return {
        "ask_location": f"Where is the {name}?" if name else "Where?",
        "ask_time": (f"When is the {name} open?" if t != "APPOINTMENT" else "When is my appointment?") if name else "When?",
        "ask_who": f"Who can help me at the {name}?" if name else "Who?",
        "ask_info": f"What can I do at the {name}?" if name else "What?",
        "request_help": "Can you help me?", "greet": "Hello.", "farewell": "Goodbye.", "thank": "Thank you.",
        "apologize": "Sorry.", "affirm": "Yes.", "deny": "No.", "request_repeat": "Could you repeat that?",
        "not_understood": "I don't understand.", "understood": "I understand.",
    }.get(act, "(unclear)")


def kb_answer(intent, history):
    act, t = intent["act"], intent["topic"]
    if t in KB:
        k = KB[t]
        name = k["name"][0].upper() + k["name"][1:]
        if act == "ask_time":
            return f"{name} - hours: {k['hours']}."
        if act in ("ask_who", "ask_info"):
            return f"{name} ({k['location']}): {k['who']}."
        return f"{name} - location: {k['location']}. Hours: {k['hours']}."
    last = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), None)
    return {
        "greet": "Hello! How can I help you today?",
        "farewell": "Goodbye, have a good day!",
        "thank": "You're welcome!",
        "apologize": "No problem at all. How can I help?",
        "request_repeat": f"Of course: {last}" if last else "Of course. What would you like me to repeat?",
        "not_understood": "Sorry, let me try again. What are you looking for?",
        "understood": "Great. Anything else?",
        "affirm": "Okay.", "deny": "Okay, what did you mean?",
        "request_help": "Of course. I can help with the library, classrooms, bathrooms, the student office, "
                        "the doctor, appointments, food and water. What do you need?",
        "ask_location": "Where are you looking for? I can help with the library, classrooms, bathrooms, student office, cafeteria, doctor, or water fountains.",
        "ask_time": "What service or facility are you asking hours for? (e.g. library, student office, or cafeteria)",
        "ask_who": "Who are you looking for? I can help you find campus staff, the student office, or the doctor.",
        "ask_info": "What do you need help with? I can help with campus locations, hours, or appointments.",
    }.get(act, "Sorry, I didn't understand. Could you sign that again?")


def rules_reply(mode, glosses, history, gate):
    """Offline stand-in for the LLM. Deterministic, clearly labelled, NOT an LLM result."""
    pending = pending_clarification(history) if mode != "direct" else None
    intent = parse_intent(glosses, history_topic(history), pending)
    if mode in ("ours", "ours_nogate") and gate["decision"] in ("llm_decides", "clarify"):
        # For actual near-ties (top-2 margin < MIN_MARGIN): clarify
        is_near_tie = (gate.get("margin", 1.0) < MIN_MARGIN)
        # For low confidence on place/service topics not yet mentioned in history: clarify
        is_unsupported_topic = (intent["topic"] != "NONE" and intent["topic"] != history_topic(history))
        is_standalone_act = intent["act"] in ("greet", "thank", "apologize", "request_help", "farewell",
                                              "affirm", "deny", "ask_location", "ask_time", "ask_who", "ask_info")
        if is_near_tie or (gate["decision"] == "clarify" and is_unsupported_topic and not is_standalone_act):
            return {"interpretation": intent_to_text(intent), "intent": intent,
                    "needs_clarification": True, "response": None}
        elif gate["decision"] == "clarify" and not is_standalone_act and not (intent["topic"] != "NONE" and intent["topic"] == history_topic(history)):
            return {"interpretation": intent_to_text(intent), "intent": intent,
                    "needs_clarification": True, "response": None}
    return {"interpretation": intent_to_text(intent), "intent": intent,
            "needs_clarification": False, "response": kb_answer(intent, history)}


# --------------------------------------------------------------------------- prompts
SCHEMA = (
    '{"intent": {"act": <one of ' + ", ".join(ACTS) + '>, "topic": <one of ' + ", ".join(TOPICS) + '>}, '
    '"interpretation": <the student\'s message as natural English>, '
    '"needs_clarification": <true|false>, "response": <your reply to the student>}'
)
KB_TEXT = json.dumps(KB, indent=1)

BASE_SYSTEM = f"""You are SignBridge, a help-desk assistant at a university. You talk with Deaf and hard-of-hearing students who sign in American Sign Language (ASL). A sign recognizer turns their signing into ASL glosses (uppercase words in ASL order, no English grammar). Raised eyebrows mark a yes/no or wh-question.

Your job for each new turn:
1. Work out what the student means, using the conversation so far to resolve short follow-ups (e.g. "WHEN" after talking about the library means "When is the library open?"; a single place sign after you asked "X or Y?" answers that question).
2. Reply helpfully and briefly, using ONLY the facts in the knowledge base. If the knowledge base has no answer, say so.

Knowledge base:
{KB_TEXT}

Answer with one JSON object and nothing else:
{SCHEMA}
"intent.topic" is the place/service the message is about (NONE if none). If you set needs_clarification to true, "response" must be a short question to the student."""

SYSTEM_B = BASE_SYSTEM

SYSTEM_C = BASE_SYSTEM + f"""

The recognizer can be wrong. For every sign you get its confidence and the alternative candidates. Policy:
- high confidence (>= {HIGH:.2f}): trust the top candidate.
- medium confidence ({LOW:.2f}-{HIGH:.2f}): respond only if the conversation clearly supports the top candidate over the alternatives. Otherwise set needs_clarification to true and ask which of the candidates the student meant.
- low confidence (< {LOW:.2f}) or two candidates almost tied: ask for clarification.
Never answer a question about a place or service the student may not have signed."""


def user_prompt(mode, segs, cues, history):
    cue_txt = ", ".join(k.replace("_", " ") for k, v in (cues or {}).items() if v) or "none"
    if mode == "llm":
        signs = " ".join(s["gloss"] for s in segs)
        body = f"Signed glosses: {signs}\nNon-manual cues: {cue_txt}"
    else:
        lines = []
        for i, s in enumerate(segs, 1):
            alts = ", ".join(f"{g} {p:.2f}" for g, p in _alts(s)) or "none"
            lines.append(f"{i}. {s['gloss']} (confidence {s['conf']:.2f}; alternatives: {alts})")
        overall = min(s["conf"] for s in segs)
        body = ("Recognized signs, in order:\n" + "\n".join(lines) +
                f"\nOverall confidence: {overall:.2f} ({tier(overall)})\nNon-manual cues: {cue_txt}")
    return f"Conversation so far:\n{transcript(history)}\n\nNew turn from the student:\n{body}"


# --------------------------------------------------------------------------- Team Competition Metadata
MEMBER_1_APPROACH = {
    "member": "Member 1",
    "model": "Qwen2.5-3B-Instruct",
    "technique": "Plain Prompting (Zero-Shot Baseline)",
    "role": "Foundational Anchor Baseline to beat",
    "description": "Zero-shot prompt directly encoding knowledge base and confidence-aware policy. Runs locally on normal laptops without GPU or training overhead.",
    "is_user_approach": True,
}

TEAM_APPROACHES = {
    "member_1": {
        "member": "Member 1 (Ours)",
        "model": "Qwen2.5-3B-Instruct",
        "technique": "Plain Prompting (Zero-Shot)",
        "role": "Foundational Anchor Baseline to beat",
        "best_fit_reason": "Repo default and easiest option. Establishes the clean baseline that the other three must beat without extra code or training.",
    },
    "member_2": {
        "member": "Member 2",
        "model": "Llama-3.2-3B-Instruct",
        "technique": "Chain-of-Thought (CoT)",
        "role": "Reasoning Cogitator",
        "best_fit_reason": "Model first writes step-by-step reasoning before emitting JSON. Tests whether reasoning cuts wrong answers on medium_wrong and near_tie.",
    },
    "member_3": {
        "member": "Member 3",
        "model": "Gemma (2B or 4B Instruct)",
        "technique": "LoRA Fine-Tuning (Hugging Face PEFT)",
        "role": "Domain Parameter-Efficient Fine-Tuner",
        "best_fit_reason": "Trained on (history, signs, confidence) -> JSON intent pairs on Colab T4 GPU and exported to Ollama.",
    },
    "member_4": {
        "member": "Member 4",
        "model": "Phi-3.5-mini or SmolLM2-1.7B",
        "technique": "Few-Shot Prompting with Retrieved Examples",
        "role": "Dynamic In-Context Example Retriever",
        "best_fit_reason": "Picks the most similar solved examples per turn. Cheap, needs no training, and shows what a small model can do with the right examples.",
    },
}


# --------------------------------------------------------------------------- backends
class Backend:
    name = "none"
    is_llm = True
    def chat(self, system, user):  # -> raw text
        raise NotImplementedError


class OllamaBackend(Backend):
    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
        self.name = f"Member 1: Qwen2.5-3B-Instruct (Ollama)"
    def available(self):
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=1).json().get("models", [])
            return any(m.get("name", "").startswith(self.model) for m in tags)
        except Exception:
            return False
    def chat(self, system, user):
        r = requests.post(f"{self.url}/api/chat", timeout=120, json={
            "model": self.model, "stream": False, "format": "json", "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAIBackend(Backend):
    def __init__(self):
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.key = os.getenv("OPENAI_API_KEY", "")
        self.name = f"openai:{self.model}"
    def chat(self, system, user):
        r = requests.post(f"{self.base}/chat/completions", timeout=60,
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "temperature": 0,
                                "response_format": {"type": "json_object"},
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicBackend(Backend):
    def __init__(self):
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        self.key = os.getenv("ANTHROPIC_API_KEY", "")
        self.name = f"anthropic:{self.model}"
    def chat(self, system, user):
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                          headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                          json={"model": self.model, "max_tokens": 600, "temperature": 0, "system": system,
                                "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


class RulesBackend(Backend):
    name = "Member 1: Qwen2.5-3B-Instruct (Plain Prompting - rules fallback)"
    is_llm = False


_backend_cache = {}


def get_backend(choice=None):
    choice = (choice or os.getenv("SIGNBRIDGE_LLM", "auto")).lower()
    if choice in _backend_cache:
        return _backend_cache[choice]
    if choice == "ollama":
        b = OllamaBackend()
    elif choice == "openai":
        b = OpenAIBackend()
    elif choice == "anthropic":
        b = AnthropicBackend()
    elif choice == "rules":
        b = RulesBackend()
    else:  # auto
        if os.getenv("ANTHROPIC_API_KEY"):
            b = AnthropicBackend()
        elif os.getenv("OPENAI_API_KEY"):
            b = OpenAIBackend()
        else:
            o = OllamaBackend()
            b = o if o.available() else RulesBackend()
    _backend_cache[choice] = b
    return b


def parse_llm_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(m.group(0) if m else text)
    intent = obj.get("intent") or {}
    act = str(intent.get("act", "unknown")).strip().lower()
    topic = str(intent.get("topic", "NONE")).strip().upper()
    return {
        "interpretation": str(obj.get("interpretation", "")).strip(),
        "intent": {"act": act if act in ACTS else "unknown", "topic": topic if topic in TOPICS else "NONE"},
        "needs_clarification": bool(obj.get("needs_clarification", False)),
        "response": str(obj.get("response", "")).strip(),
    }


# --------------------------------------------------------------------------- main entry point
def ask(history, segs, cues=None, mode="ours", backend=None):
    """Interpret one signed turn.

    history: list of stored turns (see app.py); segs: [{"gloss", "conf", "alts": [[gloss, p], ...]}]
    Returns a dict with interpretation, intent, needs_clarification, response, source, decision.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of {MODES}")
    glosses = [s["gloss"].upper() for s in segs]
    gate = assess(segs)
    t0 = time.time()

    def done(res, source):
        res.update(source=source, decision=gate["decision"], confidence=gate["confidence"],
                   tier=gate["tier"], latency_s=round(time.time() - t0, 3))
        if res["needs_clarification"]:
            if not res.get("response"):
                res["response"] = clarification_question(segs, gate["weakest"])
            if segs and gate["weakest"] is not None and gate["weakest"] < len(segs):
                s = segs[gate["weakest"]]
                cands = [s["gloss"]] + [g for g, p in _alts(s) if p >= 0.05 and g != s["gloss"]]
                res["clarification_options"] = [c.upper() for c in cands[:3]]
        return res

    if mode == "direct":
        text = " ".join(WORD.get(g, g.lower()) for g in glosses).capitalize() + "."
        return done({"interpretation": text, "intent": parse_intent(glosses),
                     "needs_clarification": False,
                     "response": "(Model A: direct sign-to-text, no reply)"}, "direct")

    if not segs:
        return done({"interpretation": "", "intent": {"act": "unknown", "topic": "NONE"},
                     "needs_clarification": True, "response": None}, "no signs")

    if mode == "ours" and gate["decision"] == "clarify":
        return done({"interpretation": "Unclear", "intent": {"act": "unknown", "topic": "NONE"},
                     "needs_clarification": True, "response": None}, "confidence gate")

    b = backend or get_backend()
    if not b.is_llm:
        return done(rules_reply(mode, glosses, history, gate), b.name)

    system = SYSTEM_B if mode == "llm" else SYSTEM_C
    try:
        res = parse_llm_json(b.chat(system, user_prompt(mode, segs, cues, history)))
    except Exception as e:  # network error, bad JSON, ... -> counted as a failure in evaluation
        return done({"interpretation": "", "intent": {"act": "unknown", "topic": "NONE"},
                     "needs_clarification": False, "response": f"(LLM error: {type(e).__name__}: {e})",
                     "error": True}, b.name)
    if mode == "ours" and gate["decision"] == "respond" and res["needs_clarification"]:
        pass  # high confidence but the LLM still asked: allowed, and visible in the evaluation
    return done(res, b.name)


def record_turn(history, segs, res):
    """Append a user turn + assistant turn to the history in the format ask() expects."""
    signs = " ".join(s["gloss"] for s in segs)
    user_txt = f"[{signs}]" + (f" -> \"{res['interpretation']}\"" if res.get("interpretation") not in (None, "", "Unclear") else "")
    history.append({"role": "user", "content": user_txt, "glosses": [s["gloss"] for s in segs],
                    "intent": None if res.get("needs_clarification") else res.get("intent")})
    asst = {"role": "assistant", "content": res.get("response") or "", "clarification": bool(res.get("needs_clarification"))}
    if res.get("needs_clarification"):
        asst["pending_intent"] = parse_intent([s["gloss"] for s in segs])
    history.append(asst)
    return history
