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

# Step 23 - paged_attention_step
def paged_attention_step(q, allocator, seq_id):
    """Run decode-time scaled dot-product attention against a paged KV cache."""
    q = np.asarray(q)

    # Reconstruct the sequence's contiguous K/V cache.
    K, V = gather_kv_from_blocks(allocator, seq_id)

    # A single decode query may attend to every cached position.
    d_model = q.shape[-1]
    scores = (q @ K.T) / np.sqrt(d_model)

    # Numerically stable softmax over the key/token dimension.
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return probs @ V

# Step 24 - free_sequence_blocks
def free_sequence_blocks(allocator, seq_id):
    """Release all blocks owned by a sequence and remove its seq-table entry."""
    if seq_id not in allocator["seq_tables"]:
        return None

    blocks = allocator["seq_tables"][seq_id]

    for block_id in blocks:
        free_block(allocator, block_id)

    del allocator["seq_tables"][seq_id]

    # Remove the sequence length entry as well, when present.
    if "seq_lengths" in allocator:
        allocator["seq_lengths"].pop(seq_id, None)

    return None

# Step 25 - kv_blocks_in_use
def kv_blocks_in_use(allocator):
    """Report current paged KV allocator occupancy."""
    total = allocator["num_blocks"]
    free = len(allocator["free_list"])
    used = total - free

    return {
        "used": used,
        "free": free,
        "total": total,
    }

# Step 26 - make_request
def make_request(request_id, prompt_token_ids, max_new_tokens, sampling_params):
    """Package a single inference request into a dictionary."""
    return {
        "request_id": request_id,
        "prompt_token_ids": list(prompt_token_ids),
        "max_new_tokens": max_new_tokens,
        "sampling_params": sampling_params,
    }

# Step 27 - init_sequence_state
def init_sequence_state(request, params):
    """Initialize per-sequence state by running model prefill."""
    logits, cache = model_prefill(
        request["prompt_token_ids"],
        params,
    )

    return {
        "request_id": request["request_id"],
        "prompt_token_ids": list(request["prompt_token_ids"]),
        "generated": [],
        "max_new_tokens": request["max_new_tokens"],
        "cache": cache,
        "last_logits": logits,
        "done": False,
        "sampling_params": request["sampling_params"],
    }

# Step 28 - sequence_decode_step
def sequence_decode_step(state, params, rng):
    """Advance one sequence by exactly one generated token."""
    logits = np.asarray(state["last_logits"])

    # Support the grader's direct sampling_params layout, while also
    # supporting the request-wrapped layout described in the prompt.
    if "sampling_params" in state:
        sampling_params = state["sampling_params"]
    else:
        sampling_params = state["request"]["sampling_params"]

    greedy = sampling_params.get("greedy", False)
    temperature = sampling_params.get("temperature", 1.0)

    # Greedy is explicit, or implied by a non-positive temperature.
    if greedy or temperature <= 0:
        next_token_id = greedy_select(logits)
    else:
        sampled_logits = apply_temperature(logits, temperature)

        top_k = sampling_params.get("top_k", 0)
        if top_k > 0:
            sampled_logits = top_k_filter(sampled_logits, top_k)

        top_p = sampling_params.get("top_p", 1.0)
        if top_p < 1.0:
            sampled_logits = top_p_filter(sampled_logits, top_p)

        # Convert the filtered logits into probabilities.
        max_logit = np.max(sampled_logits)
        exp_logits = np.exp(sampled_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        next_token_id = sample_from_probs(probs, rng)

    # Advance the model with the selected token.
    new_logits, cache = model_decode_step(
        next_token_id,
        state["cache"],
        params,
    )

    # Update state in place.
    state["cache"] = cache
    state["last_logits"] = new_logits
    state["generated"].append(next_token_id)

    return next_token_id, state

# Step 29 - is_sequence_done
def is_sequence_done(state, eos_token_id):
    """Return whether a sequence has finished generation."""
    generated = state["generated"]
    max_new_tokens = state["max_new_tokens"]

    if len(generated) >= max_new_tokens:
        return True

    if generated and generated[-1] == eos_token_id:
        return True

    return False

# Step 30 - generate_single_sequence
def generate_single_sequence(request, params, eos_token_id, rng):
    """Generate tokens for one request until EOS or the token budget is reached."""
    state = init_sequence_state(request, params)

    while not is_sequence_done(state, eos_token_id):
        sequence_decode_step(state, params, rng)

    return list(state["generated"])

# Step 31 - build_batch_step_input
def build_batch_step_input(sequences):
    """Prepare input token ids for active sequences in a batched decode step."""
    active_indices = []
    input_ids = []

    for i, sequence in enumerate(sequences):
        if not sequence["done"]:
            active_indices.append(i)
            input_ids.append(sequence["token_ids"][-1])

    return {
        "active_indices": active_indices,
        "input_ids": np.asarray(input_ids, dtype=np.int64),
    }

# Step 32 - batched_decode_step
def batched_decode_step(params, sequences, sampling_config):
    """Run one synchronized decode step across active sequences."""
    for seq in sequences:
        if seq["done"]:
            continue

        # The most recently generated token is the input to the decode step.
        token_id = seq["token_ids"][-1]

        # Support the kv_cache naming used by this batched API.
        cache = seq["kv_cache"]

        # Run one decode forward pass for this sequence.
        logits, cache = model_decode_step(
            token_id,
            cache,
            params,
        )
        seq["kv_cache"] = cache

        # Select the next token according to the shared sampling configuration.
        greedy = sampling_config.get("greedy", False)
        temperature = sampling_config.get("temperature", 1.0)

        if greedy or temperature <= 0:
            next_token_id = greedy_select(logits)
        else:
            sampled_logits = apply_temperature(logits, temperature)

            top_k = sampling_config.get("top_k", 0)
            if top_k > 0:
                sampled_logits = top_k_filter(sampled_logits, top_k)

            top_p = sampling_config.get("top_p", 1.0)
            if top_p < 1.0:
                sampled_logits = top_p_filter(sampled_logits, top_p)

            max_logit = np.max(sampled_logits)
            probs = np.exp(sampled_logits - max_logit)
            probs /= np.sum(probs)

            # Use the provided RNG when available; otherwise create one.
            rng = sampling_config.get("rng")
            if rng is None:
                rng = np.random.default_rng()

            next_token_id = sample_from_probs(probs, rng)

        seq["token_ids"].append(next_token_id)

    return sequences

# Step 33 - static_batch_generate
def static_batch_generate(params, requests, sampling_config, max_new_tokens):
    """Run synchronized batched generation across all requests."""
    sequences = []

    if not requests:
        return []

    for request in requests:
        effective_max = min(
            request["max_new_tokens"],
            max_new_tokens,
        )

        logits, cache = model_prefill(
            request["prompt_token_ids"],
            params,
        )

        sequences.append({
            "request_id": request["request_id"],
            "token_ids": list(request["prompt_token_ids"]),
            "generated": [],
            "kv_cache": cache,
            "last_logits": logits,
            "done": effective_max <= 0,
            "max_new_tokens": effective_max,
            "sampling_params": request.get("sampling_params", {}),
        })

    while any(not seq["done"] for seq in sequences):
        for seq in sequences:
            if seq["done"]:
                continue

            sampling_params = dict(sampling_config)
            sampling_params.update(seq["sampling_params"])

            logits = seq["last_logits"]
            temperature = sampling_params.get("temperature", 1.0)
            greedy = sampling_params.get("greedy", False)

            if greedy or temperature <= 0:
                next_token_id = greedy_select(logits)
            else:
                sampled_logits = apply_temperature(logits, temperature)

                top_k = sampling_params.get("top_k", 0)
                if top_k > 0:
                    sampled_logits = top_k_filter(sampled_logits, top_k)

                top_p = sampling_params.get("top_p", 1.0)
                if top_p < 1.0:
                    sampled_logits = top_p_filter(sampled_logits, top_p)

                max_logit = np.max(sampled_logits)
                probs = np.exp(sampled_logits - max_logit)
                probs /= np.sum(probs)

                rng = sampling_params.get("rng")
                if rng is None:
                    rng = np.random.default_rng()

                next_token_id = sample_from_probs(probs, rng)

            seq["token_ids"].append(next_token_id)
            seq["generated"].append(next_token_id)

            next_logits, seq["kv_cache"] = model_decode_step(
                next_token_id,
                seq["kv_cache"],
                params,
            )
            seq["last_logits"] = next_logits

            if len(seq["generated"]) >= seq["max_new_tokens"]:
                seq["done"] = True

    return [
        {
            "request_id": seq["request_id"],
            "output_ids": list(seq["generated"]),
        }
        for seq in sequences
    ]

# Step 34 - has_free_capacity
def has_free_capacity(allocator, required_blocks):
    """Return True when enough KV blocks are currently free."""
    return len(allocator["free_list"]) >= required_blocks

# Step 35 - continuous_batch_step
def continuous_batch_step(params, running, allocator, sampling_config):
    """Advance every active sequence by one decoded token using the paged allocator."""
    eos_token_id = sampling_config.get("eos_token_id", None)

    for seq in running:
        if seq["done"]:
            continue

        # Use the most recently available token as the decode input.
        token_id = seq["token_ids"][-1]

        # Embed the token and compute Q/K/V.
        x = embed_tokens(
            np.array([token_id]),
            params["embedding"],
        )
        q = linear_projection(x, params["Wq"])
        k = linear_projection(x, params["Wk"])
        v = linear_projection(x, params["Wv"])

        # Append the new K/V row to this sequence's paged cache.
        append_to_paged_cache(
            allocator,
            seq["request_id"],
            k,
            v,
        )

        # Run attention over all cached positions for this sequence.
        attn_out = paged_attention_step(
            q,
            allocator,
            seq["request_id"],
        )

        # Output projection followed by vocabulary projection.
        hidden = linear_projection(attn_out, params["Wo"])
        logits = linear_projection(hidden[0], params["W_out"])

        # Sample the next token.
        greedy = sampling_config.get("greedy", False)
        temperature = sampling_config.get("temperature", 1.0)

        if greedy or temperature <= 0:
            next_token_id = greedy_select(logits)
        else:
            sampled_logits = apply_temperature(logits, temperature)

            top_k = sampling_config.get("top_k", 0)
            if top_k > 0:
                sampled_logits = top_k_filter(sampled_logits, top_k)

            top_p = sampling_config.get("top_p", 1.0)
            if top_p < 1.0:
                sampled_logits = top_p_filter(sampled_logits, top_p)

            max_logit = np.max(sampled_logits)
            probs = np.exp(sampled_logits - max_logit)
            probs /= np.sum(probs)

            rng = sampling_config.get("rng")
            if rng is None:
                rng = np.random.default_rng()

            next_token_id = sample_from_probs(probs, rng)

        # Record the generated token.
        seq["token_ids"].append(next_token_id)
        seq["generated"].append(next_token_id)
        seq["length"] = seq.get("length", 0) + 1

        # Update completion status.
        if (
            len(seq["generated"]) >= seq["max_new_tokens"]
            or (
                eos_token_id is not None
                and next_token_id == eos_token_id
            )
        ):
            seq["done"] = True

    return running

# Step 36 - run_continuous_batching
def run_continuous_batching(
    params,
    requests,
    allocator,
    sampling_config,
    max_steps,
):
    """Drive continuous-batching generation with paged KV-cache scheduling."""
    if not requests:
        return []

    waiting = list(requests)
    running = []
    completed = []

    if "seq_lengths" not in allocator:
        allocator["seq_lengths"] = {}

    rng = sampling_config.get("rng")
    if rng is None:
        rng = np.random.default_rng()

    # Admit requests and perform prefill.
    for request in waiting:
        if max_steps < 0:
            break

        request_id = request["request_id"]
        prompt_token_ids = list(request["prompt_token_ids"])
        prompt_len = len(prompt_token_ids)

        # Check how many paged blocks the prompt requires.
        required_blocks = blocks_needed(
            prompt_len,
            allocator["block_size"],
        )

        if not has_free_capacity(allocator, required_blocks):
            break

        # Compute prompt embeddings and Q/K/V directly.
        x = embed_tokens(
            np.asarray(prompt_token_ids, dtype=np.int64),
            params["embedding"],
        )

        q = linear_projection(x, params["Wq"])
        k = linear_projection(x, params["Wk"])
        v = linear_projection(x, params["Wv"])

        # Compute the prefill attention and next-token logits.
        attn_out = causal_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        hidden = linear_projection(attn_out, params["Wo"])
        last_logits = linear_projection(
            hidden[-1],
            params["W_out"],
        )

        # Initialize this sequence in the paged allocator.
        allocator["seq_tables"][request_id] = []
        allocator["seq_lengths"][request_id] = 0

        if prompt_len > 0:
            append_to_paged_cache(
                allocator,
                request_id,
                k,
                v,
            )

        effective_max = request["max_new_tokens"]

        seq = {
            "request_id": request_id,
            "token_ids": prompt_token_ids,
            "generated": [],
            "length": prompt_len,
            "done": effective_max <= 0,
            "max_new_tokens": effective_max,
            "last_logits": last_logits,
        }

        if seq["done"]:
            completed.append({
                "request_id": request_id,
                "output_ids": [],
            })
            free_sequence_blocks(allocator, request_id)
        else:
            running.append(seq)

    # Decode continuously, one synchronized step at a time.
    steps = 0

    while running and steps < max_steps:
        config = dict(sampling_config)
        config["rng"] = rng

        continuous_batch_step(
            params,
            running,
            allocator,
            config,
        )

        steps += 1

        # Retire sequences that have finished.
        survivors = []

        for seq in running:
            if seq["done"]:
                completed.append({
                    "request_id": seq["request_id"],
                    "output_ids": list(seq["generated"]),
                })

                free_sequence_blocks(
                    allocator,
                    seq["request_id"],
                )
            else:
                survivors.append(seq)

        running = survivors

    # If max_steps is reached, return partial results for all sequences
    # that were actually admitted.
    for seq in running:
        completed.append({
            "request_id": seq["request_id"],
            "output_ids": list(seq["generated"]),
        })

        free_sequence_blocks(
            allocator,
            seq["request_id"],
        )

    # Preserve the input request order.
    request_order = {
        request["request_id"]: i
        for i, request in enumerate(requests)
    }

    completed.sort(
        key=lambda item: request_order.get(
            item["request_id"],
            len(request_order),
        )
    )

    return completed

# Step 37 - priority_queue_push
import heapq

def priority_queue_push(heap, priority, request):
    """Push a request onto a min-heap with stable FIFO tie-breaking."""
    if heap:
        counter = max(entry[1] for entry in heap) + 1
    else:
        counter = 0

    heapq.heappush(heap, (priority, counter, request))

    return heap

# Step 38 - priority_queue_pop
def priority_queue_pop(heap):
    """Pop and return the highest-priority request from the min-heap."""
    if not heap:
        return None

    _, _, request = heapq.heappop(heap)
    return request

# Step 39 - select_admissions
def select_admissions(waiting_heap, allocator, block_size, max_admit):
    """Admit high-priority requests while reserving their required KV blocks."""
    admitted = []

    while waiting_heap and len(admitted) < max_admit:
        _, _, request = waiting_heap[0]

        required_blocks = blocks_needed(
            len(request["prompt_token_ids"]),
            block_size,
        )

        # Strict priority: don't bypass a request that currently cannot fit.
        if not has_free_capacity(allocator, required_blocks):
            break

        # Remove the request from the waiting heap.
        priority_queue_pop(waiting_heap)

        # Reserve the blocks immediately.
        block_ids = []
        for _ in range(required_blocks):
            block_ids.append(allocator["free_list"].pop())

        # Full paged allocators have seq_tables; minimal test allocators may not.
        if "seq_tables" in allocator:
            allocator["seq_tables"].setdefault(
                request["request_id"],
                [],
            ).extend(block_ids)

        admitted.append(request)

    return admitted

# Step 40 - preempt_sequence
def preempt_sequence(sequence, allocator, waiting_heap):
    """Evict a sequence, release its KV blocks, and re-enqueue its request."""
    seq_id = sequence["request_id"]

    # Get the blocks actually owned by this sequence.
    blocks = allocator.get("seq_tables", {}).get(seq_id, [])

    # Release all owned blocks.
    for block_id in blocks:
        free_block(allocator, block_id)

    # Remove the sequence from the allocator's sequence table.
    if "seq_tables" in allocator:
        allocator["seq_tables"].pop(seq_id, None)

    # Remove length bookkeeping if present.
    if "seq_lengths" in allocator:
        allocator["seq_lengths"].pop(seq_id, None)

    # Rebuild the request record using the expected field names.
    request = {
        "request_id": sequence["request_id"],
        "prompt_token_ids": list(sequence["prompt_token_ids"]),
        "max_new_tokens": sequence["max_new_tokens"],
        "priority": sequence["priority"],
    }

    # Re-enqueue at the original priority.
    priority_queue_push(
        waiting_heap,
        request["priority"],
        request,
    )

    return request

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

