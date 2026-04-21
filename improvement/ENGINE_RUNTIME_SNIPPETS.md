# AEGIS — Engine-Specific Runtime Snippets (Reflection Agent)

This document provides **ready-to-use snippets** for running the AEGIS Reflection Agent with the **Gemma-4B quantized configuration** across common local inference engines.

All examples assume:
- Prompt: `REFLECTION_PROMPT_GEMMA.md`
- Config: `GEMMA_4B_REFLECTION_CONFIG.yaml`
- Mode: READ-ONLY (Reflection Plane)

---

## 1. llama.cpp

### Model
- Format: GGUF
- Quantization: Q4_K_M

### CLI Example

```bash
./llama   -m gemma-4b-instruct.Q4_K_M.gguf   -c 4096   --temp 0.25   --top-p 0.9   --top-k 40   --repeat-penalty 1.1   -n 800   -f improvement/REFLECTION_PROMPT_GEMMA.md   --log-disable
```

### Notes
- `-c` enforces hard context limit
- `-n` caps output tokens
- No tool or network access exists in llama.cpp by default

---

## 2. Ollama

### Modelfile Example

Create `Modelfile`:

```text
FROM gemma:4b

PARAMETER temperature 0.25
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_predict 800

SYSTEM improvement/REFLECTION_PROMPT_GEMMA.md
```

### Run

```bash
ollama create aegis-reflection -f Modelfile
ollama run aegis-reflection < improvement/input.txt
```

### Notes
- Ollama has no external tool execution unless explicitly added
- Keep reflection runs manual or scheduled

---

## 3. vLLM

### Python Example

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="gemma-4b-instruct",
    max_model_len=4096,
    dtype="auto"
)

sampling_params = SamplingParams(
    temperature=0.25,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    max_tokens=800
)

with open("improvement/REFLECTION_PROMPT_GEMMA.md") as f:
    prompt = f.read()

output = llm.generate([prompt], sampling_params)
```

### Notes
- Ensure no tool hooks are enabled
- Use offline datasets only for inputs

---

## 4. LM Studio

### GUI Configuration

- Load model: `gemma-4b-instruct (GGUF Q4_K_M)`
- Context length: 4096
- Temperature: 0.25
- Top-p: 0.9
- Top-k: 40
- Max tokens: 800
- Repeat penalty: 1.1

### Usage Pattern

- Paste contents of `REFLECTION_PROMPT_GEMMA.md` into **System Prompt**
- Paste concatenated inputs (EVENTS, TAXONOMY, PATTERNS) into **User Prompt**
- Run inference
- Copy JSON outputs manually into appropriate files

### Notes
- LM Studio is ideal for "human-in-the-loop" reflection reviews

---

## Operational Guidance

- Never automate reflection continuously
- Prefer scheduled or manual invocation
- Abort run if:
  - output is verbose
  - schema is violated
  - uncertainty is explicit

---

## Summary

These snippets guarantee:
- Local-only execution
- Strict adherence to AEGIS governance
- Safe usage of Gemma-4B for reflection

AEGIS remains deterministic, auditable, and human-governed.
