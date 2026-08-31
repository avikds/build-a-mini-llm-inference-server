"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
import numpy as np

def stable_softmax(logits):
    """Compute a numerically stable softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)

    # Subtract the maximum logit to prevent overflow in exp().
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)

    # Normalize over the last axis.
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    """Scale logits by 1 / temperature, treating non-positive temperatures as greedy."""
    if temperature <= 0:
        return logits

    return np.asarray(logits) / temperature

# Step 3 - top_k_filter
def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    logits = np.asarray(logits)

    vocab_size = logits.shape[-1]

    if k >= vocab_size:
        return logits.copy()

    if k <= 0:
        return np.full_like(logits, -np.inf)

    # Find the kth-largest value for each row.
    kth_values = np.partition(logits, -k, axis=-1)[..., -k:]

    # Keep every value greater than or equal to the kth-largest value.
    threshold = np.min(kth_values, axis=-1, keepdims=True)
    mask = logits >= threshold

    output = logits.copy()
    output[~mask] = -np.inf

    return output

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    """Keep the smallest set of tokens whose cumulative probability reaches p."""
    logits = np.asarray(logits)

    if not (0 < p <= 1):
        raise ValueError("p must be in the interval (0, 1]")

    original_shape = logits.shape

    # Treat 1D input as a batch of size 1.
    if logits.ndim == 1:
        work_logits = logits[None, :]
    elif logits.ndim == 2:
        work_logits = logits
    else:
        raise ValueError("logits must be 1D or 2D")

    # Stable softmax.
    max_logits = np.max(work_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(work_logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Sort probabilities from largest to smallest.
    sorted_indices = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)

    # Cumulative probability in descending order.
    cumulative_probs = np.cumsum(sorted_probs, axis=-1)

    # Keep tokens until cumulative probability reaches p.
    keep_sorted = cumulative_probs <= p

    # Always keep the first token that crosses the threshold.
    crossing_index = np.argmax(cumulative_probs >= p, axis=-1)
    rows = np.arange(work_logits.shape[0])
    keep_sorted[rows, crossing_index] = True

    # Map the mask back to the original vocabulary order.
    keep_mask = np.zeros_like(keep_sorted, dtype=bool)
    np.put_along_axis(
        keep_mask,
        sorted_indices,
        keep_sorted,
        axis=-1,
    )

    output = work_logits.copy()
    output[~keep_mask] = -np.inf

    # Restore the original shape.
    if len(original_shape) == 1:
        return output[0]

    return output

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    """Draw a single token id from a categorical distribution."""
    probs = np.asarray(probs)

    return int(rng.choice(len(probs), p=probs))

# Step 6 - greedy_select
def greedy_select(logits):
    """Return the index of the maximum logit; ties go to the lowest index."""
    logits = np.asarray(logits)

    return int(np.argmax(logits))

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    """Build a tiny character-level vocabulary."""
    # Preserve special-token order while removing duplicates.
    specials = list(dict.fromkeys(special_tokens))

    # Collect all unique characters from the corpus.
    unique_chars = set()
    for text in corpus:
        unique_chars.update(text)

    # Special tokens occupy the lowest ids; characters follow in sorted order.
    tokens = specials + sorted(unique_chars - set(specials))

    token_to_id = {token: idx for idx, token in enumerate(tokens)}

    return {
        "token_to_id": token_to_id,
        "id_to_token": tokens,
    }

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    """Encode a raw string into a flat list of token ids."""
    token_to_id = vocab["token_to_id"]

    token_ids = []

    if add_bos and "<bos>" in token_to_id:
        token_ids.append(token_to_id["<bos>"])

    unk_id = token_to_id.get("<unk>")

    for char in text:
        if char in token_to_id:
            token_ids.append(token_to_id[char])
        elif unk_id is not None:
            token_ids.append(unk_id)

    return token_ids

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    """Convert token ids back into a string."""
    id_to_token = vocab["id_to_token"]

    tokens = []

    for token_id in token_ids:
        token = id_to_token[int(token_id)]

        if skip_special and token.startswith("<") and token.endswith(">"):
            continue

        tokens.append(token)

    return "".join(tokens)

# Step 10 - embed_tokens
def embed_tokens(token_ids, embedding_matrix):
    """Return embedding vectors corresponding to each token id."""
    token_ids = np.asarray(token_ids)

    return embedding_matrix[token_ids]

# Step 11 - linear_projection
def linear_projection(x, weight, bias=None):
    """Apply an affine transformation: y = x @ weight + bias."""
    y = np.asarray(x) @ np.asarray(weight)

    if bias is not None:
        y = y + np.asarray(bias)

    return y

# Step 12 - init_kv_cache
def init_kv_cache(max_seq_len, d_model):
    """Allocate a zero-initialized contiguous KV cache for a single sequence."""
    return {
        "K": np.zeros((max_seq_len, d_model), dtype=np.float32),
        "V": np.zeros((max_seq_len, d_model), dtype=np.float32),
        "length": 0,
    }

# Step 13 - append_kv
def append_kv(cache, k_new, v_new):
    """Append new key/value rows to a contiguous KV cache in place."""
    k_new = np.asarray(k_new)
    v_new = np.asarray(v_new)

    start = cache["length"]
    t = k_new.shape[0]
    end = start + t

    cache["K"][start:end] = k_new
    cache["V"][start:end] = v_new
    cache["length"] = end

    return cache

# Step 14 - causal_attention
def causal_attention(q, k, v, is_causal=True):
    """Compute scaled dot-product attention with an optional causal mask."""
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)

    tq, d = q.shape
    tk = k.shape[0]

    # Scaled attention scores.
    scores = (q @ k.T) / np.sqrt(d)

    if is_causal:
        # Query position i may attend to keys j <= i + (Tk - Tq).
        offset = tk - tq
        query_positions = np.arange(tq)[:, None]
        key_positions = np.arange(tk)[None, :]

        mask = key_positions > (query_positions + offset)
        scores = scores.copy()
        scores[mask] = -np.inf

    # Stable row-wise softmax.
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return probs @ v

# Step 15 - model_prefill
def model_prefill(token_ids, params):
    """Run the prefill pass for a tiny single-layer transformer."""
    token_ids = np.asarray(token_ids)

    # Embed the input tokens: (T,) -> (T, D)
    x = embed_tokens(token_ids, params["embedding"])

    # Project to queries, keys, and values.
    q = linear_projection(x, params["Wq"])
    k = linear_projection(x, params["Wk"])
    v = linear_projection(x, params["Wv"])

    # Initialize and populate the KV cache.
    d_model = params["embedding"].shape[1]
    cache = init_kv_cache(params["max_seq_len"], d_model)
    append_kv(cache, k, v)

    # Causal self-attention over the full prompt.
    attn_out = causal_attention(q, cache["K"][:cache["length"]], cache["V"][:cache["length"]])

    # Output projection of the attention result.
    hidden = linear_projection(attn_out, params["Wo"])

    # Project only the final position to vocabulary logits.
    logits = linear_projection(hidden[-1], params["W_out"])

    return logits, cache

# Step 16 - model_decode_step
def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    # Embed the incoming token.
    x = embed_tokens(np.array([token_id]), params["embedding"])

    # Compute query, key, and value.
    q = linear_projection(x, params["Wq"])
    k = linear_projection(x, params["Wk"])
    v = linear_projection(x, params["Wv"])

    # Append the new K/V row to the existing cache.
    append_kv(cache, k, v)

    # The decode query attends to all cached positions.
    attn_out = causal_attention(
        q,
        cache["K"][:cache["length"]],
        cache["V"][:cache["length"]],
        is_causal=True,
    )

    # Output projection.
    hidden = linear_projection(attn_out, params["Wo"])

    # Vocabulary logits from the single decode position.
    logits = linear_projection(hidden[0], params["W_out"])

    return logits, cache

# Step 17 - blocks_needed
def blocks_needed(num_tokens, block_size):
    """Return the number of fixed-size blocks needed for num_tokens tokens."""
    if num_tokens == 0:
        return 0

    return (num_tokens + block_size - 1) // block_size

# Step 18 - init_block_allocator
def init_block_allocator(num_blocks, block_size, d_model):
    """Initialize a paged KV cache block allocator."""
    return {
        "K_blocks": np.zeros(
            (num_blocks, block_size, d_model),
            dtype=np.float32,
        ),
        "V_blocks": np.zeros(
            (num_blocks, block_size, d_model),
            dtype=np.float32,
        ),
        "free_list": list(range(num_blocks)),
        "block_size": block_size,
        "num_blocks": num_blocks,
        "d_model": d_model,
        "seq_tables": {},
    }

# Step 19 - allocate_block
def allocate_block(allocator, seq_id):
    """Allocate one KV block to a sequence."""
    if not allocator["free_list"]:
        raise RuntimeError("Out of KV cache blocks")

    # LIFO allocation: take the last available block.
    block_id = allocator["free_list"].pop()

    # Lazily create the sequence's block table.
    if seq_id not in allocator["seq_tables"]:
        allocator["seq_tables"][seq_id] = []

    allocator["seq_tables"][seq_id].append(block_id)

    return block_id

# Step 20 - free_block
def free_block(allocator, block_id):
    """Return a block id to the allocator's free list."""
    allocator["free_list"].append(block_id)
    return None

# Step 21 - append_to_paged_cache
def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    """Write t new K/V rows into the sequence's paged blocks, allocating as needed."""
    k_new = np.asarray(k_new)
    v_new = np.asarray(v_new)

    t = k_new.shape[0]
    block_size = allocator["block_size"]

    # Initialize per-sequence length tracking.
    if "seq_lengths" not in allocator:
        allocator["seq_lengths"] = {}

    current_length = allocator["seq_lengths"].get(seq_id, 0)

    # Lazily initialize the sequence's block table.
    if seq_id not in allocator["seq_tables"]:
        allocator["seq_tables"][seq_id] = []

    blocks = allocator["seq_tables"][seq_id]

    # Allocate enough blocks to hold the new positions.
    required_blocks = (current_length + t + block_size - 1) // block_size

    while len(blocks) < required_blocks:
        allocate_block(allocator, seq_id)

    # Write each new row into its logical paged position.
    for i in range(t):
        logical_pos = current_length + i
        block_idx = logical_pos // block_size
        offset = logical_pos % block_size
        block_id = blocks[block_idx]

        allocator["K_blocks"][block_id, offset] = k_new[i]
        allocator["V_blocks"][block_id, offset] = v_new[i]

    # Update the sequence length.
    allocator["seq_lengths"][seq_id] = current_length + t

    return None

# Step 22 - gather_kv_from_blocks
def gather_kv_from_blocks(allocator, seq_id):
    """Reconstruct contiguous K and V arrays from a sequence's paged blocks."""
    blocks = allocator["seq_tables"][seq_id]
    length = allocator["seq_lengths"][seq_id]
    block_size = allocator["block_size"]
    d_model = allocator["d_model"]

    K = np.empty((length, d_model), dtype=np.float32)
    V = np.empty((length, d_model), dtype=np.float32)

    for pos in range(length):
        block_idx = pos // block_size
        offset = pos % block_size
        block_id = blocks[block_idx]

        K[pos] = allocator["K_blocks"][block_id, offset]
        V[pos] = allocator["V_blocks"][block_id, offset]

    return K, V

# Step 23 - paged_attention_step (not yet solved)
# TODO: implement

# Step 24 - free_sequence_blocks (not yet solved)
# TODO: implement

# Step 25 - kv_blocks_in_use (not yet solved)
# TODO: implement

# Step 26 - make_request (not yet solved)
# TODO: implement

# Step 27 - init_sequence_state (not yet solved)
# TODO: implement

# Step 28 - sequence_decode_step (not yet solved)
# TODO: implement

# Step 29 - is_sequence_done (not yet solved)
# TODO: implement

# Step 30 - generate_single_sequence (not yet solved)
# TODO: implement

# Step 31 - build_batch_step_input (not yet solved)
# TODO: implement

# Step 32 - batched_decode_step (not yet solved)
# TODO: implement

# Step 33 - static_batch_generate (not yet solved)
# TODO: implement

# Step 34 - has_free_capacity (not yet solved)
# TODO: implement

# Step 35 - continuous_batch_step (not yet solved)
# TODO: implement

# Step 36 - run_continuous_batching (not yet solved)
# TODO: implement

# Step 37 - priority_queue_push (not yet solved)
# TODO: implement

# Step 38 - priority_queue_pop (not yet solved)
# TODO: implement

# Step 39 - select_admissions (not yet solved)
# TODO: implement

# Step 40 - preempt_sequence (not yet solved)
# TODO: implement

# Step 41 - schedule_step (not yet solved)
# TODO: implement

# Step 42 - format_stream_chunk (not yet solved)
# TODO: implement

# Step 43 - submit_request (not yet solved)
# TODO: implement

# Step 44 - drive_until_complete (not yet solved)
# TODO: implement

# Step 45 - collect_request_output (not yet solved)
# TODO: implement

# Step 46 - build_completion_response (not yet solved)
# TODO: implement

# Step 47 - time_to_first_token (not yet solved)
# TODO: implement

# Step 48 - inter_token_latency (not yet solved)
# TODO: implement

# Step 49 - aggregate_throughput (not yet solved)
# TODO: implement

# Step 50 - latency_percentiles (not yet solved)
# TODO: implement

# Step 51 - run_throughput_latency_benchmark (not yet solved)
# TODO: implement

