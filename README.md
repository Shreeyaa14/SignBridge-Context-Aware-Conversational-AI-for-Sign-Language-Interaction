# SignBridge - Context-Aware Conversational AI-for-Sign-Language-Interaction

Most service counters have no one who can sign. For a Deaf person, that usually
means pen and paper, lip-reading, or waiting for an interpreter. SignBridge is a
small step towards closing that gap: a Deaf person signs to a webcam, and the
hearing person at the counter reads (or hears) what they said and can answer
straight back.

## How a conversation works

1. The Deaf person signs a short question to the webcam, for example WHERE + LIBRARY.
2. SignBridge tracks the hands and face and works out which signs were made,
   along with how sure it is about each one.
3. If it is unsure, it asks: "Did you mean library or bathroom?" The signer taps
   the right answer instead of signing everything again.
4. If it is sure, a language model turns the signs into a proper sentence
   ("Where is the library?") and drafts a reply.
5. The hearing person reads the sentence or listens to it, then answers by typing
   or speaking. Their answer appears on screen for the signer.

## How it works technically

- **Sign recognition:** MediaPipe Holistic in the browser (hands, upper body,
  eyebrows, mouth) → a bidirectional GRU with attention that picks from 23
  isolated ASL signs and returns its top 3 guesses with calibrated confidence.
- **Language layer:** Qwen2.5-3B-Instruct, run locally with Ollama, zero-shot.
  It gets the top 3 guesses, their confidence and the last six turns of the
  conversation, and returns JSON: intent, interpretation and reply.
- **Safety check in code:** if the weakest sign is below 0.50 confidence, or the
  top two guesses are within 0.10 of each other, the system asks a clarifying
  question before the LLM is called at all.
- **Interface:** FastAPI backend, a single web page, speech input and
  text-to-speech output, and a 1–5 rating on every reply.
- **Evaluation:** 208 scripted conversations with recognition errors added on
  purpose, scored on task success, wrong-answer rate, clarification and
  over-asking.

## Why it matters

Small mistakes in an interpreted conversation are easy to miss. A system that
answers confidently on a misread sign can send someone to the wrong place, or
worse. SignBridge is built around one rule: when in doubt, ask. Everything runs
locally, so camera footage never leaves the machine.

## What it is not

It is not an ASL translator. It knows 23 isolated signs, with a pause between
each, in one help-desk setting. It cannot read full ASL sentences or grammar,
it has not yet been tested with Deaf users, and it does not replace a
professional interpreter for anything important.
