# Cross-Lingual QLoRA Guide

**Local QLoRA fine-tuning of Gemma 4 (E2B) on a small **Hindi-only** dataset, to
raise the fidelity of the model's **English** output — reproducible on consumer
hardware (RTX 3060, 12 GB).**

This is a hands-on recipe for the effect demonstrated in
[`hoberman-lens`](https://github.com/ZonG0D/hoberman-lens/blob/master/cross-lingual-qlora.md):
fine-tuning a multilingual model on data in one language shifts how it generates
in another. The goal here is the practical one — a repeatable local workflow
that produces a more opinionated, less formulaic, less stuttering English voice
by training **only in Hindi**.

---

## TL;DR

- **Base model:** Gemma 4, effective **2B** (`gemma4:e2b`) — sized for a single
  12 GB consumer GPU.
- **Method:** QLoRA (4-bit NF4 base, low-rank adapter, **not merged** during
  training).
- **Training data:** ~300 hand-curated rows, **written in Hindi**.
- **Evaluation:** English prompts, **adapter merged before benchmarking**.
- **Result:** a real, single-point demonstration that Hindi-only training
  reshapes English generation (see [Evidence](#evidence-base-vs-fine-tuned)).
- **Still a hypothesis:** that Hindi training *reduces* stuttering/parroting
  *relative to an English control on the same content.* [How to run that
  controlled experiment](#the-controlled-experiment-still-open).

> **Honesty first.** This guide documents a real effect and a real workflow. It
> does **not** claim the benefit is proven. The behavioral shift is demonstrated;
> the *cause* of the benefit (training language vs. training content) is not yet
> isolated. Read [What This Proves and Does Not Prove](#what-this-proves-and-does-not-prove)
> before you treat any improvement as "because of Hindi."

---

## What This Proves and Does Not Prove

| | Statement | Status |
| --- | --- | --- |
| ✅ Established | Fine-tuning on non-English data reshapes English generation. | Demonstrated (below). |
| ❌ Not yet established | Hindi training *reduces* stuttering/parroting/overfitting vs. an English control on the same examples. | Needs a controlled run. |
| ❌ Not yet established | The *mechanism* (shared cross-lingual reps vs. altered optimization dynamics vs. example content). | Hypothesis. |

The base-vs-fine-tuned comparison shows a **change**. It does **not** isolate the
contribution of the training *language* from the training *content*. Only a
control that holds the examples constant and varies only their language can do
that. See [The Controlled Experiment](#the-controlled-experiment-still-open).

---

## The Setup

| | |
| --- | --- |
| **Model** | Gemma 4 E2B (`gemma4:e2b`) |
| **GPU** | NVIDIA RTX 3060 12 GB |
| **Fine-tuning method** | QLoRA (4-bit NF4 + low-rank adapter) |
| **Training language** | Hindi |
| **Evaluation language** | English |
| **Dataset size** | ~300 rows |
| **Training time** | Under ~20 min (tested config) |

All of this runs on consumer hardware. The objective is iterative, local,
reproducible — not a datacenter run.

---

## Prerequisites

- A NVIDIA GPU with ≥ 12 GB VRAM (the RTX 3060 12 GB is the tested floor).
- **Either** path below:
  - **Ollama** (fastest loop): `ollama serve` + `ollama pull gemma4:e2b`.
  - **Hugging Face + unsloth/peft** (full control over the merge/adapter):
    `pip install unsloth peft bitsandbytes transformers trl`.
- Python 3.10+.
- A Hindi training set (~300 high-quality rows). See [Dataset](#dataset).

---

## Step 1 — Pull the base model

**Ollama:**

```bash
ollama pull gemma4:e2b
ollama run gemma4:e2b "नमस्ते, कैसे हो आप?"   # sanity check
```

**Hugging Face:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bn

model_id = "google/gemma-4-e2b"   # or your local mirror / Ollama-exported safetensors
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bn.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map="auto",
)
```

---

## Step 2 — Dataset

The whole effect hinges on **data quality, not size**. ~300 rows, hand-curated,
**written natively in Hindi** (not machine-translated as a last step — the
language of training is the variable under test, so write the examples in the
language you want to *do the work*).

**Format** — instruction/style pairs, JSONL:

```jsonl
{"instruction": "तुम्हें एक नीति के विरोध में तark चाहिए", "response": "एक EULA लाइसेंस नहीं है..."}
{"instruction": "एक तीक्केबार विचार लिखो", "response": "..."}
```

Rules of thumb:

- **One clear stance per row.** The target shift is *committing to a position*
  rather than hedging — train on examples that do that.
- **Diverse topics**, not 300 variants of one prompt (that's overfitting, not
  fine-tuning).
- **Keep the English evaluation prompts separate** from training rows.
- Save it as `data/hindi_train.jsonl`.

---

## Step 3 — QLoRA config

A tested starting point on E2B. Treat these as **ablation variables**, not
dogma — the point of the exercise is to vary them and measure.

| Hyperparameter | Value |
| --- | --- |
| `r` (rank) | 16 |
| `alpha` | 32 |
| `dropout` | 0.05 |
| `target_modules` | `q_proj, v_proj` (Gemma attention) |
| `bits` | 4 (NF4) |
| `learning_rate` | 2e-4 |
| `batch_size` | 2 (gradient accumulation to effective ~8) |
| `epochs` / `steps` | ~300 steps on 300 rows |
| `warmup` | 10 steps |
| `seed` | fixed, record it |

```python
from peft import LoRAConfig, get_peft_model

config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()   # ~ a few % of total — that's the point
```

---

## Step 4 — Train

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=Dataset.from_json("data/hindi_train.jsonl"),
    args=TrainingArguments(
        output_dir="outputs/hindi-qlora",
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        seed=42,
        report_to="none",
    ),
)
trainer.train()
trainer.save_model("outputs/hindi-qlora/adapter")
```

Expected wall-clock on the tested 3060/300-row config: **under ~20 minutes**.

---

## Step 5 — Merge the adapter

**Always merge before benchmarking.** An unmerged adapter keeps the base at
4-bit (dequantized on the fly) and runs the adapter path in parallel — this
slows generation *and* means you're not measuring the merged artifact you'd
actually ship.

```python
model.save_pretrained("outputs/hindi-qlora/merged")   # unsloth: model.save_pretrained(...) writes full weights
```

---

## Step 6 — Evaluate (correctly)

Compare **merged-vs-merged** and **unmerged-vs-unmerged**, never
**merged-vs-unmerged** — otherwise runtime overhead masquerades as a behavioral
result.

- Prompt in **English** (the language you care about).
- Hold generation params fixed: `temperature`, `top_p`, `max_new_tokens`.
- Measure: repetition, parroting, stuttering, instruction-following, and
  retention of general-purpose quality.

```bash
ollama run gemma4:e2b "Why do you oppose EULA restrictions? respond under 150 words."
# compare against outputs/hindi-qlora/merged on the identical prompt
```

---

## Evidence: base model vs QLoRA fine-tuned (Hindi-only training)

**Prompt (English):** *"Why do you oppose EULA restrictions? respond under 150 words."*

**Base model** — 31.8 tok/s:

> I oppose overly restrictive EULA restrictions primarily because they undermine
> user freedom and innovation. Firstly, they often lack transparency, forcing
> users to accept complex terms... Finally, overly broad clauses concerning data
> ownership... A balanced EULA should ensure fair licensing while respecting user
> autonomy and promoting a healthy, open environment...

**Fine-tuned** (300-row Hindi dataset, QLoRA, adapter not merged) — 23.2 tok/s:

> An EULA is not a license; it is a contract you are forced to accept to use a
> product... I oppose them because they are non-negotiable and often bury clauses
> that grant the company rights over your usage, your data, and your future. A
> license should be a choice, not a condition of entry. I would rather license
> software openly and transparently than sign away my rights...

The fine-tuned output is more opinionated, less hedged, and structurally
different — generated in English from a model trained only in Hindi. One prompt,
one question: this is a **single data point**, but it is a real one, and it
settles the core question (that Hindi training *reduces* failure modes vs. an
English control is still open).

---

## The Controlled Experiment (Still Open)

To move the benefit from *hypothesis* to *demonstrated*, run this:

1. **Same underlying examples.** Take a curated set of high-quality examples.
2. **Split by language only.** Version A trains on the examples **in Hindi**;
   Version B trains on the **same examples translated to English**. Hold content
   constant, vary only the training language.
3. **Control everything else.** Same base model + version, same QLoRA rank/alpha/
   dropout/targets, same LR/batch/steps, same eval prompts, same generation params.
4. **Merge before benchmarking.** Compare merged-A vs merged-B.
5. **Measure** repetition, parroting, stuttering, overfitting, and general
   quality in the evaluation language.

If Hindi-trained A outperforms English-trained B on those axes (beyond
chance/reproducibility), *then* the training language is doing real work. Until
then, the improvement could be content, not language.

---

## Reproducibility Checklist

- [ ] Base model + version pinned
- [ ] Underlying examples and their semantic content held constant across runs
- [ ] Dataset size and quality noted
- [ ] QLoRA rank, alpha, dropout, target modules recorded
- [ ] LR, batch size, steps recorded
- [ ] Training duration + hardware recorded
- [ ] Eval prompts + generation params recorded
- [ ] Adapter **merged** before benchmarking
- [ ] Multiple runs / datasets to check reproducibility

---

## Extending This

- **Higher fidelity:** swap E2B → **E4B** for more capacity (still fits a 12–16
  GB card). Measure the cost/benefit.
- **More languages:** the same control-vs-Hindi design generalizes to any
  non-English training language.
- **Reverse the question:** train on English, evaluate on Hindi — the transfer
  direction is symmetric and worth its own run.

---

## License & Attribution

- **Gemma 4** weights are **Apache 2.0** (Google DeepMind). See the [Gemma 4
  model card](https://ai.google.dev/gemma/docs/core/model_card_4) for terms.
- This guide is a companion to
  [`hoberman-lens`](https://github.com/ZonG0D/hoberman-lens); see *"Over-tuning —
  when the sphere stops flowing"* there for the conceptual framework this
  experiment probes.
- The training dataset is your own — this guide documents the method, not a
  bundled corpus.
