from smolagents import CodeAgent, InferenceClientModel, Tool
from gradio_client import Client


class ImageGenerationTool(Tool):
    name = "image_generator"
    description = "Generate an image from a text prompt using FLUX."

    inputs = {
        "prompt": {"type": "string", "description": "The image description/prompt"}
    }

    output_type = "string"

    def __init__(self):
        super().__init__()
        self.client = Client(
            "black-forest-labs/FLUX.1-schnell",
            token=None,  # Uses your Hugging Face authentication
        )

    def forward(self, prompt: str):
        result = self.client.predict(prompt=prompt, api_name="/infer")
        return str(result)


image_generation_tool = ImageGenerationTool()

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "Improve this prompt, then generate an image of it.",
    additional_args={
        "user_prompt": "A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala"
    },
)
