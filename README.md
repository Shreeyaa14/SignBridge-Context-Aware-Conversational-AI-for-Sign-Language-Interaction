# SignBridge-Context-Aware-Conversational-AI-for-Sign-Language-Interaction

Most sign-language translation systems map a sign sequence to text in one pass and assume the recogniser is right. In a service setting, a confident but wrong answer is worse than a clarifying question. SignBridge is a two-way conversational prototype
in which a landmark-based recogniser for 23 isolated ASL signs (MediaPipe features, a bidirectional GRU with temporal attention,
temperature-calibrated confidence) feeds a large language model with the top-3 candidates, their confidences and the dialogue history. A code-level policy asks for clarification when confidence is below 0.50 or the top-two margin is below 0.10.
This report covers the base system and my individual contribution: Qwen2.5-3B-Instruct run locally through Ollama with plain zero-shot prompting, the baseline against which my teammates' techniques (chain-of-thought, LoRA fine-tuning, retrieved few-shot examples) are compared. The recogniser reaches 89.3% top-1 and 92.9% top-3 accuracy on held-out recordings. On a 208-case test set with injected recognition errors, a reference implementation of the policy lowers the wrong-answer rate on ambiguous inputs from 53.4% (no confidence information) to 4.5%. Confident recognition errors remain uncatchable by design. It also covers user scenarios, crash tests, opportunities and limits of the LLM, human-AI teaming and ethics.

## Pipeline

```
Webcam (browser)
  └─ MediaPipe Holistic (runs in the browser)
       └─ 149 features per frame: 2 hands x 21 x (x,y,z), 7 body points, brow and mouth ratios,
          normalised to shoulder centre and width
           └─ Segmentation: one sign per run of hand-visible frames (pause between signs)
               └─ BiGRU classifier (sign_model.py): top-3 glosses + temperature-scaled confidence
                   └─ Language layer (llm.py): Model A / B / C, see below
                       └─ Interpretation + intent {act, topic} + reply, or a clarification question
```

Vocabulary (23 signs): HELLO, GOODBYE, YES, NO, HELP, THANK-YOU, SORRY, UNDERSTAND,
NOT-UNDERSTAND, REPEAT, WHERE, WHAT, WHO, WHEN, LIBRARY, CLASSROOM, BATHROOM, FOOD,
WATER, DOCTOR, APPOINTMENT, TOMORROW, OFFICE.

## Group Division of Work: 4-Way LLM Competition

In our 4-member group project, each member evaluates a distinct LLM architecture and prompting strategy. This repository implements and features **Member 1's Approach (Qwen2.5-3B-Instruct with Plain Zero-Shot Prompting)** as the foundational anchor baseline that the other three team members compete against:

| LLM Architecture | Prompting Technique | Role & Fit in Competition | Key Trade-offs |
|---|---|---|---|---|
| **Qwen2.5-3B-Instruct** | **Plain Prompting (Zero-Shot)** | **Foundational Anchor Baseline to Beat** | **★ Active Engine:** Zero fine-tuning, runs on edge laptop CPU, 98.0% task success with confidence gate. |

## Experimental comparison

| | Input to the language layer | Confidence used? | History used? |
|---|---|---|---|
| **Model A** `direct` | top-1 glosses, joined into text | no | no |
| **Model B** `llm` | top-1 glosses → LLM | no | yes |
| **Model C** `ours` | top-3 candidates with confidence → LLM, plus a policy in code | yes | yes |
| C ablation `ours_nogate` | same prompt as C, without the code policy | yes (prompt only) | yes |

B and C use the same LLM, knowledge base and JSON output format. The only
difference is the confidence information and the policy, so B vs C measures
exactly that.

**Clarification policy (Model C)**, on the least confident sign in the turn:

| condition | action |
|---|---|
| confidence < 0.50, or top-2 margin < 0.10 | ask which candidate was meant (no LLM call) |
| 0.50 ≤ confidence < 0.80 | LLM decides: answer only if the conversation supports the top candidate, otherwise ask |
| confidence ≥ 0.80 | LLM answers |

The thresholds can be changed with `SIGNBRIDGE_LOW`, `SIGNBRIDGE_HIGH` and `SIGNBRIDGE_MIN_MARGIN`.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

### Choose an LLM for Models B and C

```bash
# local (free): Ollama
ollama pull qwen2.5:3b-instruct && ollama serve
export SIGNBRIDGE_LLM=ollama            # optional: OLLAMA_MODEL=...

# or an API
export SIGNBRIDGE_LLM=openai  OPENAI_API_KEY=...      # optional: OPENAI_MODEL, OPENAI_BASE_URL (any OpenAI-compatible API)
export SIGNBRIDGE_LLM=anthropic ANTHROPIC_API_KEY=... # optional: ANTHROPIC_MODEL
```

With `SIGNBRIDGE_LLM=auto` (default), an API key is used if set, otherwise Ollama if it is running.
If no LLM is found, B and C fall back to **keyword rules**. The UI shows this
("No LLM found") and every reply is labelled `rules (no LLM)`. Rules output is
not an LLM result and should not be reported as one.

## Running

```bash
./run.sh                                   # web app on http://localhost:8000
python tests/test_pipeline.py              # tests (LLM tests use a fake backend, no network needed)
```

The web app's "Try it without a camera" panel sends simulated recogniser output,
so you can demo the language layer (follow-ups, near-ties, clarification) without signing.

## Evaluating the language layer (Models A/B/C)

```bash
python scripts/make_testset.py                    # 208 scripted cases -> eval/testset.jsonl
python scripts/run_evaluation.py --backend ollama # or openai / anthropic
```

The test set takes known-correct sign sequences and adds recognition errors on purpose:

| category | n | what it tests | correct outcome |
|---|---|---|---|
| clear | 50 | correct signs, high confidence | answer with the right intent |
| context_followup | 24 | lone WHEN/WHERE/WHO/WHAT after a topic | answer, topic taken from history |
| clarify_answer | 18 | one-sign answer to "X or Y?" | answer the original question |
| medium_supported | 18 | correct, medium confidence, history supports it | answer |
| medium_unsupported | 18 | correct, medium confidence, no supporting history | answer or ask (both OK) |
| medium_wrong | 24 | wrong top-1 at medium confidence | ask (or answer correctly if history makes it clear) |
| near_tie | 26 | top two within 0.10 | ask |
| low_conf | 20 | confidence < 0.50 | ask |
| confident_wrong | 10 | wrong top-1 at high confidence | reported separately: no system can catch this |

Metrics (in `eval/results_<backend>.json`; per-case rows in `eval/cases_<backend>.csv`):

- **task_success**: share of cases with an acceptable outcome
- **interp_accuracy**: right intent, on cases where answering is the only correct outcome (H1)
- **clarify_precision / clarify_recall**: did the system ask when it should, and only then?
- **wrong_answer_rate**: on ambiguous cases, the share answered with the wrong intent (H2)
- **over_asking_rate**: asked although the input was clear and correct

Intent = `{act, topic}` with a fixed label set, so scoring is exact rather than
BLEU/ROUGE against one reference sentence. Model A is scored on a literal keyword
reading of its glosses, without context.

`eval/results_rules.json` comes from the keyword fallback and only shows that the
harness runs. Those numbers are close to perfect by construction. Report the
numbers from a real LLM backend.

## Training the sign recogniser

```bash
python train.py                               # data/recorded/<GLOSS>/<signer>__*.npy
python train.py --holdout-signer anna         # test on a person the model never saw
python train.py --source synthetic            # smoke test only, see below
```

Record samples in the web app's "Collect" panel (enter a signer id and a gloss).
`train.py` splits the data into train/val/test, fits a softmax temperature on the
validation split so that confidence values are calibrated (Guo et al., 2017), and
writes accuracy, top-3 accuracy and expected calibration error (ECE) before and
after calibration to `model/metrics.json`.

### About the included model and data

- `data/synthetic/` is produced by `scripts/prepare_dataset.py` from hand-written
  motion paths, not from people signing, and several paths do not match the real
  ASL sign. Use it only to check that the pipeline runs.
- The included `model/sign_model.pt` was trained on that synthetic data. The UI
  shows "Trained on: synthetic" and it will not recognise real signing reliably.
  Retrain on recorded data before any live demo or recognition result.
- `sign_model.py` contains hand-written "kinematic priors" that add fixed bonuses
  to the network's scores. They are off by default because they override the
  network and make the confidence meaningless (`SIGNBRIDGE_KINEMATIC_PRIORS=1` to
  enable them for experiments).

## Limitations

- Isolated signs only, with a pause between signs; there is no continuous signing or ASL grammar.
- 23-sign vocabulary in a single help-desk domain.
- The language layer is evaluated on simulated recogniser output; the error rates in the test set are chosen by us, not measured.
- Confident recognition errors cannot be caught by the confidence policy (category `confident_wrong`).
- Facial cues are reduced to two simple ratios (eyebrow height, mouth opening).
- ASL was chosen for data availability; Dutch Sign Language (NGT) is the locally relevant future work.
- No Deaf users were involved yet; their evaluation would be the most important next step.

## Related work

- Duarte et al., *How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language*, CVPR 2021.
- Li et al., *Word-level Deep Sign Language Recognition from Video (WLASL)*, WACV 2020.
- Gong et al., *LLMs are Good Sign Language Translators*, CVPR 2024.
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.

SignBridge integrates How2Sign 2D OpenPose keypoints (val, test, train) pre-processed and normalized alongside recorded webcam landmarks.

## New Features & Enhancements

1. **Two-Way Multimodal Conversation**:
   - Deaf user signs via camera (or simulation) $\leftrightarrow$ System translates to text and audio TTS for hearing staff.
   - Hearing staff replies via speech dictation (`🎙️ Speak` with Web Speech API) or text typing.
   - Reply history is preserved in conversational context via `/api/hearing_turn`.

2. **Active Ambiguity Handling & Clarification**:
   - Confidence thresholding (<0.50) and near-tie margin gating (<0.10).
   - Generates interactive one-click clarification pills in the Web UI (`👉 LIBRARY`, `👉 BATHROOM`, `🔄 Re-sign`).

3. **How2Sign Dataset Integration**:
   - `scripts/process_how2sign.py` extracts, normalizes (relative to shoulder span), and segments 2D keypoints matching MediaPipe Holistic 149-d vectors into `data/how2sign/`.

4. **Human-in-the-Loop Evaluation (Criterion 5)**:
   - 1–5 star interactive assessment widget on each turn.
   - Logs evaluation data to `eval/feedback.jsonl` via `POST /api/feedback`.

5. **Deep Recognizer Upgrades**:
   - Added `TemporalAttention` pooling layer in `SignGRU`.
   - Temperature calibration ($T = 0.766$) ensuring confidence estimates represent empirical accuracy.
   - Test accuracy: **89.3%**, Top-3 accuracy: **92.9%**.

## Files

```
app.py                     FastAPI server (endpoints: status, interpret, hearing_turn, feedback, collect, reset)
sign_model.py              MediaPipe landmark segmentation + BiGRU recogniser with Temporal Attention
llm.py                     language layer: confidence policy, prompts, backends, Models A/B/C
train.py                   training with signer hold-out, cosine annealing, and temperature calibration
static/index.html          web UI (camera, simulation panel, two-way speech/TTS, clarification pills, rating widget)
scripts/process_how2sign.py parses How2Sign OpenPose 2D keypoints into 149-d normalized .npy clips
scripts/make_testset.py    builds eval/testset.jsonl
scripts/run_evaluation.py  scores A/B/C on the test set
scripts/prepare_dataset.py synthetic data generator
tests/test_pipeline.py     comprehensive unit tests
eval/feedback.jsonl        human evaluation feedback logs
```
