"""
Utility functions for interacting with OpenAI Chat Completions.

This module is used to execute prompt-based experiments for
the gastrointestinal oncology tumor board study.

IMPORTANT:
- No patient data is stored in this file.
- API keys and Assistant IDs must be provided by the user.
- This code is intended for research purposes only.
"""


from openai import OpenAI

# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

# Insert your OpenAI project API key here or load it from environment variables
API_KEY_PROJECT = 'YOUR API KEY'

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY_PROJECT)


# =============================================================================
# CHAT COMPLETION (STANDARD CHAT API)
# =============================================================================

def chatgpt_chat_completion(prompt_text: str, model: str) -> str:
    """
    Send a single-turn prompt to an OpenAI chat completion model.

    This function is used for:
    - Simple request (no retrieval) configurations
    - Custom RAG configurations where retrieved context
      is injected directly into the prompt

    Parameters
    ----------
    prompt_text : str
        Fully formatted prompt string sent to the model.
    model : str
        Model identifier (e.g. "gpt-5-2025-08-07").

    Returns
    -------
    str
        Model-generated response text.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        top_p=1.0,
    )

    # Extract and return the assistant's response text
    return response.choices[0].message.content.strip()

