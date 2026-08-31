# Hugging Face Agents

Small experiments from the Hugging Face Agents course.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the Hugging Face Inference API needs authentication, configure your token locally instead of committing it:

```bash
export HF_TOKEN="your-token-here"
```

## Run

```bash
cd smolAgent
python smolagent.py "What time is it in Chennai?"
python smolagent.py "Generate an image of a white crow"
```

Image answers are saved to `smolAgent/generated_answer.png`.
