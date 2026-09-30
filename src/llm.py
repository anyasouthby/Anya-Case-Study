
from mlx_lm import load, generate


MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"


class LocalLLM:
    """Wrapper around a locally loaded MLX language model."""

    def __init__(self):
        print("Loading language model...")
        self.model, self.tokenizer = load(MODEL_NAME)
        print("Language model loaded.")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using the already-loaded model."""

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        return output.strip()

