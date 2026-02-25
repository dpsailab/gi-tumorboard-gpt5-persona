"""
Run a selected experimental tumor board framework on a patient case.

This script allows runtime selection of:
    - Model version
    - Whether to use a rewritten case
    - Experimental framework (1–5)

Frameworks:
    1  = Standard Multidisciplinary Tumor Board
    2  = Multi-Expert Deliberation with Consensus Synthesis
    3  = Surgical Oncologist (single specialist)
    4  = Medical Oncologist (single specialist)
    5  = Radiation Oncologist (single specialist)

Designed for scientific reproducibility and benchmarking.
"""

# =============================================================================
# Imports
# =============================================================================

import os
from tumorboard_frameworks import get_prompts_for_framework
from openai_client import chatgpt_chat_completion


# =============================================================================
# Path Handling (Reproducible Repository Structure)
# =============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Model Selection (Runtime Control for Experimental Comparisons)
# =============================================================================

MODEL_NAME = "gpt-5-2025-08-07"


# =============================================================================
# Framework Selection (Core Experimental Variable)
# =============================================================================

print("\nSelect framework:")
print("1) Standard Multidisciplinary Tumor Board")
print("2) Multi-Expert Deliberation with Consensus")
print("3) Surgical Oncologist")
print("4) Medical Oncologist")
print("5) Radiation Oncologist")

framework_choice = input("Enter choice [1–5]: ").strip()

FRAMEWORK_MAPPING = {
    "1": "framework_1",
    "2": "framework_2",
    "3": "surgeon",
    "4": "medical_oncologist",
    "5": "radiation_oncologist",
}

if framework_choice not in FRAMEWORK_MAPPING:
    raise ValueError("Invalid framework selection. Choose 1–5.")

framework_type = FRAMEWORK_MAPPING[framework_choice]


# =============================================================================
# Case Selection (Original vs Rewritten)
# =============================================================================


case_txt_path = os.path.abspath(
    os.path.join(current_dir, '..', 'data', 'dummy_patients', 'example_case_de.txt')
)

with open(case_txt_path, 'r', encoding='utf-8') as f:
    case_text = f.read().strip()


# =============================================================================
# Prompt Construction
# =============================================================================

prompts = get_prompts_for_framework(
    case_text=case_text,
    framework_type=framework_type
)

system_prompt = prompts["system_prompt"]
user_prompt = prompts["user_prompt"]

# =============================================================================
# Model Execution
# =============================================================================

# Combine system and user prompts into a single message
full_prompt = f"{system_prompt}\n\n{user_prompt}"

print("\n=== PROMPT: ===\n",full_prompt)

print('\nrunning...')

response = chatgpt_chat_completion(
    full_prompt,
    model=MODEL_NAME
)


# =============================================================================
# Output (Transparent Experimental Logging)
# =============================================================================

print("\n==============================")
print("MODEL:", MODEL_NAME)
print("FRAMEWORK:", framework_type)
print("==============================")

print("\n=== Case Text ===")
print(case_text)

print("\n=== Model Output ===")
print(response)