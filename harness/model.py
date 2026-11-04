# Talks to a local Ollama model.
# One function: send the messages (and tools), get back the reply and timing numbers.

import time
import ollama


def chat(model, messages, tools, seed, temperature):
    start = time.time()
    response = ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
        think=False,
        options={"seed": seed, "temperature": temperature},
    )
    seconds = time.time() - start

    stats = {
        "seconds": round(seconds, 2),
        "prompt_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
    }
    return response.message, stats


# Removes the model from memory. Called after every task, because Ollama's MLX runner
# keeps growing its memory cache across tasks (seen: qwen3.5:0.8b-mlx grew 5 GB -> 10 GB).
def unload(model):
    ollama.generate(model=model, prompt="", keep_alive=0)


# Memory the model is using right now, in GB (0 if it is not loaded).
def memory_gb(model):
    for running in ollama.ps().models:
        if running.model == model or running.name == model:
            return running.size / 1e9
    return 0.0
