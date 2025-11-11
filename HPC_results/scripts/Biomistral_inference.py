from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os
from datetime import datetime

# ---- paths (EDIT if yours differ) ----
MODEL_DIR  = "/home/absz/Seminars/BioMistral/pretrained_model"
INPUT_FILE = "/home/absz/Seminars/BioMistral/data/raw/generated_questions_labels.txt"
OUTPUT_FILE = "/home/absz/Seminars/BioMistral/data/result/biomistral_plain_answers.txt"
# --------------------------------------

MAX_NEW = 48  # keep modest
MIN_NEW = 8   # <-- DEBUG: ensures we get *something* (remove later if you want)

def log(msg): print(f"[{datetime.now().isoformat()}] - {msg}", flush=True)

# ---- Load model + tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device).eval()

log(f"Loaded model on: {next(model.parameters()).device}")

# ---- Read questions ----
questions = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split(" ||| ")
        q = parts[0].split("QUESTION: ", 1)[1].strip()
        questions.append(q)
log(f"Loaded {len(questions)} questions.")

# ---- Prompt builders ----
def build_plain_prompt(q: str) -> str:
    # A tight, non-chat prompt with a clear anchor
    return (
        "You classify clinical questions using best available evidence.\n"
        "Allowed labels (choose one): REFUTED, SUPPORTED, NOT ENOUGH INFORMATION.\n"
        "Return exactly ONE line: LABEL: <label> ||| <brief explanation>\n\n"
        "Examples:\n"
        "QUESTION: Do probiotics cure type 1 diabetes?\n"
        "ANSWER: LABEL: REFUTED ||| No evidence of cure; trials show no reversal of autoimmunity.\n"
        "QUESTION: Do high-dose vitamin C eradicate all cancers?\n"
        "ANSWER: LABEL: REFUTED ||| RCTs do not show curative effect; evidence is insufficient for eradication.\n"
        "QUESTION: Do non-pharmacological interventions reduce pain during endotracheal suctioning in preterm infants?\n"
        "ANSWER: LABEL: SUPPORTED ||| RCTs show reduced pain scores with facilitated tucking and containment.\n"
        "QUESTION: Does omega-3 supplementation prevent all strokes in adults?\n"
        "ANSWER: LABEL: NOT ENOUGH INFORMATION ||| Mixed results; prevention effect not established across populations.\n\n"
        f"QUESTION: {q}\n"
        "LABEL: "
    )

def encode_with_chat_template(q: str):
    # Put the instruction inside the user message; no 'system' role.
    user_prompt = (
        "You classify clinical questions using best available evidence.\n"
        "Allowed labels (choose one): REFUTED, SUPPORTED, NOT ENOUGH INFORMATION.\n"
        "Return exactly ONE line: LABEL: <label> ||| <brief explanation>\n\n"
        "Examples:\n"
        "QUESTION: Do probiotics cure type 1 diabetes?\n"
        "ANSWER: LABEL: REFUTED ||| No evidence of cure; trials show no reversal of autoimmunity.\n"
        "QUESTION: Do high-dose vitamin C eradicate all cancers?\n"
        "ANSWER: LABEL: REFUTED ||| RCTs do not show curative effect; evidence is insufficient for eradication.\n"
        "QUESTION: Do non-pharmacological interventions reduce pain during endotracheal suctioning in preterm infants?\n"
        "ANSWER: LABEL: SUPPORTED ||| RCTs show reduced pain scores with facilitated tucking and containment.\n"
        "QUESTION: Does omega-3 supplementation prevent all strokes in adults?\n"
        "ANSWER: LABEL: NOT ENOUGH INFORMATION ||| Mixed results; prevention effect not established across populations.\n\n"
        f"QUESTION: {q}\n"
        "LABEL: "
    )
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,   # appends the assistant header the template expects
        return_tensors="pt"
    ).to(device)

use_chat = hasattr(tokenizer, "apply_chat_template")
if use_chat:
    log("Chat-template active")

# ---- Generate one-by-one (no batching) ----
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
log(f"Writing to {OUTPUT_FILE}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out, torch.inference_mode():
    for i, q in enumerate(questions, 1):
        if use_chat:
            enc_ids = encode_with_chat_template(q)
            prompt_len = enc_ids.shape[1]
            gen_kwargs = {"input_ids": enc_ids}
        else:
            prompt = build_plain_prompt(q)
            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]
            gen_kwargs = enc

        out_ids = model.generate(
            **gen_kwargs,
            max_new_tokens=MAX_NEW,
            min_new_tokens=MIN_NEW,        # <-- DEBUG: ensures non-empty continuation
            do_sample=False,               # deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Start with NO extra constraints. Re-add later if desired:
            # no_repeat_ngram_size=3,
            # repetition_penalty=1.05,
            # bad_words_ids=tokenizer(["QUESTION:"], add_special_tokens=False).input_ids,
        )

        new_tokens = out_ids[:, prompt_len:]
        # 1) Debug view (first 3 only): keep specials so we can see EOS etc.
        if i <= 3:
            dbg = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
            log(f"RAW#{i} (repr, keep specials): {repr(dbg)}")

        # 2) Actual text to save: full continuation, specials removed
        text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        # If the model still outputs nothing (very unlikely with MIN_NEW), give a safe fallback
        if not text.strip():
            text = "LABEL: NOT ENOUGH INFORMATION ||| (empty output)"

        # Save the WHOLE continuation (your request)
        out.write(text.rstrip() + "\n")
        out.flush()

        if i % 100 == 0 or i == len(questions):
            log(f"Processed {i}/{len(questions)}")

log("Done.")
