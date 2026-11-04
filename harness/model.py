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
