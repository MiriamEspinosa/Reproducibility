from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from datetime import datetime

# ---- paths (EDIT if yours differ) ----
MODEL_DIR  = "/home/absz/Seminars/Qwen/pretrained_model"
INPUT_FILE = "/home/absz/Seminars/Qwen/data/raw/generated_questions_labels.txt"
OUTPUT_FILE = "/home/absz/Seminars/Qwen/data/result/qwen_answers.txt"
# --------------------------------------

# variables
BATCH_SIZE = 1          # tune: 4–16 on V100 32GB should be fine
MAX_NEW    = 32
TEMP       = 0.0

# helper function
def log(msg):
    print(f"[{datetime.now().isoformat()}] - {msg}", flush=True)

# 1) Load local model + tokenizer
log(f"Loading Tokenizer..")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # simple fix for padding
log("Tokenizer loaded successfully!")

log("Loading model..")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
model.eval()
log("Model loaded successfully!")
# check which device is being used
for name, param in model.named_parameters():
    log(f"{name}: {param.device}")
    break  # only need the first one

# 2) Read questions (your parsing, unchanged)
questions = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n").split(" ||| ")
        q = line[0].split("QUESTION: ")[1].strip()
        questions.append(q)
log(f"Loaded {len(questions)} questions.")
# 3) Simple prompt builder (plain string, no chat format)
def build_prompt(q):
    return (
        "You are a helpful AI Assistant that classifies clinical questions using best available evidence.\n"
        "Allowed labels: SUPPORTED, REFUTED, NOT ENOUGH INFORMATION.\n"
        "Output exactly ONE line: 'LABEL: <label> ||| <brief explanation>'\n"
        "Do not include any other text.\n\n"
        "Examples:\n"
        "QUESTION: Do non-pharmacological interventions reduce pain during endotracheal suctioning in preterm infants? ANSWER: \n"
        "LABEL: SUPPORTED ||| RCTs show reduced pain scores with facilitated tucking and containment.\n"
        "QUESTION: Do probiotics cure type 1 diabetes? ANSWER: \n"
        "LABEL: REFUTED ||| No evidence of cure; trials show no reversal of autoimmunity.\n"
        "QUESTION: Does omega-3 supplementation prevent all strokes in adults? ANSWER: \n"
        "LABEL: NOT ENOUGH INFORMATION ||| Mixed results; prevention effect not established across populations.\n\n"
        f"QUESTION: {q}\n"
        "ANSWER: "
    )
# 4) Generate answers 
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Build all prompts up front
prompts = [build_prompt(q) for q in questions]

log(f"Loaded {len(prompts)} prompts; writing to {OUTPUT_FILE}")

log("Generating answer..")
with open(OUTPUT_FILE, "w", encoding="utf-8") as out, torch.inference_mode():
    total = len(prompts)
    bad_words = ["Human:", "QUESTION:"]
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    for start in range(0, total, BATCH_SIZE):
        batch_prompts = prompts[start: start+BATCH_SIZE]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # move tensors to the same device as the model
        dev = next(model.parameters()).device
        enc = {k: v.to(model.device) for k, v in enc.items()}
        outputs = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,           # use your variables
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            bad_words_ids=bad_words_ids,
        )

        # decode ONLY the newly generated tokens (not the prompt)
        new_tokens = outputs[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # one line per answer, as specified in your prompt
        for text in decoded:
            out.write(text.replace("\n", " ||| ").strip() + "\n")

        out.flush()  # so you can tail the file while it runs

        done = min(start + BATCH_SIZE, total)
        if done % 50 == 0 or done == total:
            log(f"Processed {done}/{total}")
            
            
log(f"Saved to {OUTPUT_FILE}")