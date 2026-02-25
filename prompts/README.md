# Prompts Module – GPT-5 Tumor Board Frameworks

This folder contains the scripts used to generate experimental outputs with GPT-5 for the gastrointestinal oncology tumor board simulation study.

The purpose of this module is **methodological transparency and reproducibility**.  
It allows researchers to run controlled prompt-based experiments using dummy patient cases.

⚠️ This folder does **not** contain real patient data.  
⚠️ API keys must be provided by the user.

---

# Overview of Scripts

This folder contains three core scripts:

| Script | Purpose |
|--------|----------|
| `openai_client.py` | Handles communication with the OpenAI API |
| `prompt_templates.py` | Defines all experimental prompt frameworks |
| `run_framework_experiment.py` | Runs a selected framework on a dummy case |

---

# 1️⃣ `openai_client.py`

### Purpose
Provides a minimal wrapper around the OpenAI Chat Completions API.

### Key Function

```python
chatgpt_chat_completion(prompt_text: str, model: str) -> str
```

### Behavior

Sends a single-turn prompt

Uses:
top_p = 1.0

Returns:
The model-generated output as plain text

Configuration Required
Insert your OpenAI API key:
```python
API_KEY_PROJECT = 'YOUR API KEY'
```
or load it securely from environment variables.

---

# 2️⃣ prompt_templates.py

This module defines the five experimental frameworks used in the GPT-5 tumor board study.

No patient data is included in this file.

## Framework Overview

| Framework | Description                             | Output Style                     |
| --------- | --------------------------------------- | -------------------------------- |
| 1         | Standard multidisciplinary tumor board  | 1–2 sentence German decision     |
| 2         | Multi-expert self-consistency reasoning | Structured reasoning + consensus |
| 3         | Surgical oncologist perspective         | 1–2 sentence German decision     |
| 4         | Medical oncologist perspective          | 1–2 sentence German decision     |
| 5         | Radiation oncologist perspective        | 1–2 sentence German decision     |

### Framework 1 – Standard Tumor Board

The model acts as a complete multidisciplinary tumor board and must produce a clear treatment decision.

Characteristics:

No external retrieval

No multi-step reasoning output

Output: concise German recommendation

### Framework 2 – Multi-Expert Deliberation (Self-Consistency)

The model generates three independent reasoning paths:

Surgeon

Medical Oncologist

Radiation Oncologist

It then synthesizes a final tumor board consensus.

Structured Output Format:

```plain_text
1. <surgeon>...</surgeon>
2. <onco>...</onco>
3. <radioonco>...</radioonco>
4. <tumorboard>...</tumorboard>
```

This framework explicitly enforces internal deliberation before consensus generation.

---

### Frameworks 3–5 – Single Specialist Role Prompts

The model assumes the perspective of one specialty only:

Surgical oncologist

Medical oncologist

Radiation oncologist

The model must:

Make a clear treatment decision

Not suggest further tumor board discussion

Answer in German (1–2 sentences)

Explain its reasoning briefly

---

## Prompt Construction

All prompts are generated via:
```python
get_prompts_for_framework(case_text, framework_type)
```
Returns:
````python
{
    "system_prompt": "...",
    "user_prompt": "..."
}
````
In the execution script, these are concatenated into a single message before being sent to the API.

---

# 3️⃣ run_framework_experiment.py

This script executes a selected framework on a dummy patient case.

What It Does

1. Loads a dummy German patient case from:
```code
data/dummy_patients/example_case_de.txt
```
2. Asks the user to select one of five frameworks:
````code
1) Standard Multidisciplinary Tumor Board
2) Multi-Expert Deliberation with Consensus
3) Surgical Oncologist
4) Medical Oncologist
5) Radiation Oncologist
````
3. Constructs the appropriate prompt using prompt_templates.py
4. Combines system and user prompts into a single message:

````python
full_prompt = f"{system_prompt}\n\n{user_prompt}"
````
5. Sends the prompt to GPT-5:
````python
chatgpt_chat_completion(full_prompt, model=MODEL_NAME)
````
6. Prints:
- Model name
- Framework used
- Input case
- Model output

---

## How to Run

From the prompts/ directory:
````bash
python run_framework_experiment.py
````

You will be prompted to select a framework.

---

## Model Configuration

Current default model in the script (used in the experiments):
````python
MODEL_NAME = "gpt-5-2025-08-07"
````

This can be modified for comparative experiments.

## Experimental Characteristics

- Single-turn prompting
- No external retrieval
- No assistant file storage
- Deterministic setup (top_p = 1.0)

This module isolates prompt-engineering effects only, independent of retrieval augmentation.

## Intended Use

This module is designed for:

- Prompt engineering experiments
- Specialty role simulation studies
- Self-consistency evaluation
- Tumor board reasoning analysis
- Methodological transparency in scientific publications

It is NOT intended for clinical use!!

## Reproducibility Notes

To ensure reproducibility:
- Fix the model version
- Keep prompt templates unchanged
- Log framework selection
- Store raw outputs before post-processing

## Citation

If you use these prompt frameworks in academic work, please cite the associated study.

## Disclaimer

This repository is for research purposes only.
Outputs generated by large language models must not be used for real clinical decision-making.
