# Talks to an Ollama server: the local one by default, or another one given with --host.
#
# Safety limits on every model call:
#   NUM_CTX            fixed context size, so memory cannot grow with long prompts
#   MAX_OUTPUT_TOKENS  a reply stops after this many tokens. Without it, qwen once generated
#                      for 7 min 15 s in a single call (Phase 1 run, 2026-11-04). Normal replies are
#                      far shorter (qwen max 329, gemma max 3,016, Refiner max 888 tokens).
#   TIMEOUT_SECONDS    a call that takes longer is an infrastructure problem (server stuck,
#                      server unreachable). It raises ModelError and the run STOPS, so a broken
#                      connection is never counted as the model failing a task.

import json
import time
import urllib.request

import ollama

NUM_CTX = 32768
MAX_OUTPUT_TOKENS = 4096
TIMEOUT_SECONDS = 300
DEFAULT_HOST = "http://127.0.0.1:11434"

host = DEFAULT_HOST
client = ollama.Client(host=DEFAULT_HOST, timeout=TIMEOUT_SECONDS)

# Models that run on a different server than the default one, e.g. a refiner or embedding
# model on another Ollama server while the worker runs locally: model name -> (host, client).
model_hosts = {}


class ModelError(Exception):
    pass


def set_host(new_host):
    global host, client
    host = new_host
    client = ollama.Client(host=new_host, timeout=TIMEOUT_SECONDS)


def set_model_host(model, model_host):
    model_hosts[model] = (model_host, ollama.Client(host=model_host, timeout=TIMEOUT_SECONDS))


# The (host, client) a model's calls go to.
def server_for(model):
    if model in model_hosts:
        return model_hosts[model]
    return host, client


def chat(model, messages, tools, seed, temperature):
    model_host, model_client = server_for(model)
    start = time.time()
    try:
        response = model_client.chat(
            model=model,
            messages=messages,
            tools=tools,
            think=False,
            options={
                "seed": seed,
                "temperature": temperature,
                "num_ctx": NUM_CTX,
                "num_predict": MAX_OUTPUT_TOKENS,
            },
        )
    except Exception as error:
        raise ModelError("model call failed after " + str(round(time.time() - start)) + " s on " + model_host + ": " + repr(error)[:300])
    seconds = time.time() - start

    stats = {
        "seconds": round(seconds, 2),
        "prompt_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
        "cut_at_max_tokens": response.done_reason == "length",
    }
    return response.message, stats


# Embedding vectors for a list of texts (used by the memory database, technique #3).
def embed(model, texts):
    model_host, model_client = server_for(model)
    try:
        response = model_client.embed(model=model, input=texts)
    except Exception as error:
        raise ModelError("embed failed on " + model_host + ": " + repr(error)[:300])
    return response.embeddings


# Removes the model from memory. Called after every task, because Ollama's MLX runner
# keeps growing its memory cache across tasks (seen: qwen3.5:0.8b-mlx grew 5 GB -> 10 GB).
def unload(model):
    model_host, model_client = server_for(model)
    try:
        model_client.generate(model=model, prompt="", keep_alive=0)
    except Exception as error:
        raise ModelError("unload failed on " + model_host + ": " + repr(error)[:300])


# Memory the model is using right now, in GB (0 if it is not loaded).
def memory_gb(model):
    model_host, model_client = server_for(model)
    try:
        running_models = model_client.ps().models
    except Exception as error:
        raise ModelError("ps failed on " + model_host + ": " + repr(error)[:300])
    for running in running_models:
        if running.model == model or running.name == model:
            return running.size / 1e9
    return 0.0


# Written into each run's config.json, so every result says which server and model ran it.
def server_info(model):
    model_host, model_client = server_for(model)
    info = {"host": model_host}
    with urllib.request.urlopen(model_host + "/api/version", timeout=10) as response:
        info["ollama_version"] = json.loads(response.read())["version"]
    for item in model_client.list().models:
        if item.model == model:
            info["model_digest"] = item.digest
    return info
