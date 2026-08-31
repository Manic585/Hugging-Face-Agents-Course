from smolagents import (
    CodeAgent,
    InferenceClientModel,
    LogLevel,
    tool,
    load_tool,
)
import datetime
import pytz
import yaml
import argparse
from pathlib import Path

from final_answer import FinalAnswerTool


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The localtime is {local_time}"
    except Exception as e:
        return f"Exception in fetching the localtime of {timezone} - {e}"


BASE_DIR = Path(__file__).resolve().parent
output_path = BASE_DIR / "generated_answer.png"


def show_answer(answer):
    if hasattr(answer, "save"):
        answer.save(output_path)
        print(f"Image saved to: {output_path.resolve()}")
    else:
        print(answer)


def build_agent():
    final_answer = FinalAnswerTool()
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        max_tokens=2096,
        temperature=0.5,
        custom_role_conversions=None,
    )

    image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

    with open(BASE_DIR / "prompts.yaml", "r") as stream:
        prompt_templates = yaml.safe_load(stream)

    return CodeAgent(
        model=model,
        tools=[final_answer, get_current_time_in_timezone, image_generation_tool],
        prompt_templates=prompt_templates,
        verbosity_level=LogLevel.OFF,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What time is it in Chennai?",
        help="Prompt to send to the smolagents CodeAgent.",
    )
    args = parser.parse_args()

    agent = build_agent()
    answer = agent.run(args.prompt)
    show_answer(answer)
