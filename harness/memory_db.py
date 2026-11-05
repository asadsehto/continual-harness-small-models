# Technique #3: memory database.
# In C1 every memory entry goes into the agent's prompt, and memory is capped at 20 entries.
# With the memory database:
#   - the Refiner can store up to MAX_DB_ENTRIES entries (no 20-entry cap)
#   - at the start of each task, only the TOP_K entries most similar to the task instruction
#     are put into the prompt (cosine similarity of nomic-embed-text embeddings)

import math

from harness.model import embed

EMBED_MODEL = "nomic-embed-text:latest"
TOP_K = 5
MAX_DB_ENTRIES = 200

# text -> vector, so each memory entry is embedded only once
cache = {}


def entry_text(entry):
    return entry["title"] + ": " + entry["content"]


def cosine(a, b):
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot = dot + x * y
        norm_a = norm_a + x * x
        norm_b = norm_b + y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def retrieve(memory, query, k=TOP_K):
    # Returns the k entries most similar to the query, best first.
    if len(memory) <= k:
        return list(memory)
    # nomic-embed-text expects these prefixes for search
    missing = []
    for entry in memory:
        if entry_text(entry) not in cache:
            missing.append(entry_text(entry))
    if missing:
        vectors = embed(EMBED_MODEL, ["search_document: " + text for text in missing])
        for text, vector in zip(missing, vectors):
            cache[text] = vector
    query_vector = embed(EMBED_MODEL, ["search_query: " + query])[0]
    scored = []
    for entry in memory:
        scored.append((cosine(query_vector, cache[entry_text(entry)]), entry))
    scored.sort(key=lambda pair: -pair[0])
    return [entry for score, entry in scored[:k]]
