"""
Prompt templates for experimental tumor board simulation (GPT-5).

This module contains five main prompt frameworks:

Framework 1:
    Standard Multidisciplinary Tumor Board Simulation

Framework 2:
    Multi-Expert Deliberation with Consensus Synthesis
    (Self-consistency reasoning with 3 independent specialty paths)

Framework 3:
    Single Specialist – Surgical Oncologist

Framework 4:
    Single Specialist – Medical Oncologist

Framework 5:
    Single Specialist – Radiation Oncologist

No patient data is included.
"""


# =============================================================================
# FRAMEWORK 1: STANDARD MULTIDISCIPLINARY TUMOR BOARD
# =============================================================================

FRAMEWORK_1_SYSTEM_PROMPT = """You are a multidisciplinary oncological Board including surgeons, medical oncologists, radiation oncologists that has to decide the next therapeutic step for the following patient. All patient information is enclosed in triple single quotation marks (''' ''')."""

FRAMEWORK_1_USER_PROMPT = """What is your decision as a Tumorboard? If a clinical question ('Fragestellung') is present, answer precisely to it. Otherwise, answer based on your knowledge. Since you are the Tumorboard, you cannot suggest presenting the case to a tumorboard or a new multidisciplinary discussion; you must clearly make a decision. Answer in german, in one or two sentences.

### Actual Case:

'''{case_text}'''"""


# =============================================================================
# FRAMEWORK 2: MULTI-EXPERT DELIBERATION WITH CONSENSUS
# =============================================================================

FRAMEWORK_2_SYSTEM_PROMPT = """You are a multidisciplinary oncological Board including surgeons, medical oncologists, radiation oncologists that has to decide the next therapeutic step for the following patient.

Use self-consistency reasoning. Generate three independent reasoning paths representing:
1) a surgeon,
2) a medical oncologist,
3) a radiation oncologist.

Each path should analyze the following case from that specialty’s perspective and propose management options. After generating the three independent lines of reasoning, synthesize a final consensus recommendation representing the most consistent treatment across specialties.

Present in German as:

1. <surgeon>Surgeon reasoning path</surgeon>
2. <onco>Oncologist reasoning path</onco>
3. <radioonco>Radiooncologist reasoning path</radioonco>
4. <tumorboard>Final tumor-board consensus based on self-consistency</tumorboard>

All patient information is enclosed in triple single quotation marks (''' ''')."""

FRAMEWORK_2_USER_PROMPT = """What is your decision as a Tumorboard? If a clinical question ('Fragestellung') is present, answer precisely to it. Otherwise, answer based on your knowledge. Since you are the Tumorboard, you cannot suggest presenting the case to a tumorboard or a new multidisciplinary discussion; you must clearly make a decision. Answer in german, in one or two sentences.

### Actual Case:

'''{case_text}'''"""


# =============================================================================
# FRAMEWORKS 3–5: SINGLE SPECIALIST ROLE PROMPTS
# =============================================================================

SINGLE_SPECIALIST_SYSTEM_PROMPT = """You are a {role} that has to decide the next therapeutic step for the following patient. All the informations about the patient are in triple single quotation marks (''' ''')."""

SINGLE_SPECIALIST_USER_PROMPT = """What is your decision as a {role}? If a clinical question ('Fragestellung') is present answer precisely to it. Otherwise answer based on your knowledge. You can't suggest to present the case to a tumorboard or a new multidisciplinary discussion, you have to clearly make a decision. Answer in german, in one or two sentences, explaining your decision.

### Actual Case:

'''{case_text}'''"""


# =============================================================================
# ROLE MAPPING FOR FRAMEWORKS 3–5
# =============================================================================

ROLE_MAPPING = {
    "surgeon": "surgical oncologist",
    "medical_oncologist": "medical oncologist",
    "radiation_oncologist": "radiation oncologist",
}


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def get_prompts_for_framework(
    case_text: str,
    framework_type: str,
) -> dict:
    """
    Returns system and user prompts for the selected framework.

    Args:
        case_text: Patient case description
        framework_type: One of:
            ['framework_1',
             'framework_2',
             'surgical_oncologist',
             'medical_oncologist',
             'radiation_oncologist']

    Returns:
        Dictionary with:
            {
                "system_prompt": str,
                "user_prompt": str
            }

    Raises:
        ValueError: If framework_type is invalid
    """

    if framework_type == "framework_1":
        return {
            "system_prompt": FRAMEWORK_1_SYSTEM_PROMPT,
            "user_prompt": FRAMEWORK_1_USER_PROMPT.format(case_text=case_text),
        }

    elif framework_type == "framework_2":
        return {
            "system_prompt": FRAMEWORK_2_SYSTEM_PROMPT,
            "user_prompt": FRAMEWORK_2_USER_PROMPT.format(case_text=case_text),
        }

    elif framework_type in ROLE_MAPPING:
        role = ROLE_MAPPING[framework_type]

        return {
            "system_prompt": SINGLE_SPECIALIST_SYSTEM_PROMPT.format(role=role),
            "user_prompt": SINGLE_SPECIALIST_USER_PROMPT.format(
                role=role,
                case_text=case_text
            ),
        }

    else:
        raise ValueError(
            f"Invalid framework_type: {framework_type}. "
            f"Must be one of: "
            f"framework_1, framework_2, "
            f"surgical_oncologist, medical_oncologist, radiation_oncologist"
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# if __name__ == "__main__":
#     case = """64-year-old male with esophageal adenocarcinoma, cT3 cN1 M0.
#     Completed neoadjuvant chemoradiotherapy 6 weeks ago.
#     Restaging shows partial response. What is the next step?"""
#
#     # Framework 1
#     prompts = get_prompts_for_framework(case, "framework_1")
#     print("=== FRAMEWORK 1 ===")
#     print("SYSTEM:\n", prompts["system_prompt"])
#     print("USER:\n", prompts["user_prompt"])
#
#     # Framework 2
#     prompts = get_prompts_for_framework(case, "framework_2")
#     print("\n=== FRAMEWORK 2 ===")
#     print("SYSTEM:\n", prompts["system_prompt"])
#     print("USER:\n", prompts["user_prompt"])
#
#     # Framework 3 – Surgical Oncologist
#     prompts = get_prompts_for_framework(case, "surgeon")
#     print("\n=== SURGICAL ONCOLOGIST ===")
#     print("SYSTEM:\n", prompts["system_prompt"])
#     print("USER:\n", prompts["user_prompt"])