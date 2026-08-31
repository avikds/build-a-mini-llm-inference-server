# Build a Mini LLM Inference Server

Construct a complete LLM inference stack from scratch: sampling, tokenization, a tiny transformer with KV caching, a paged attention allocator, continuous batching with scheduling, a streaming serving API, and a throughput/latency benchmark harness. This mirrors the architecture of modern serving systems like vLLM at a digestible scale.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** stable_softmax
- [x] **2.** apply_temperature
- [x] **3.** top_k_filter
- [x] **4.** top_p_filter
- [x] **5.** sample_from_probs
- [x] **6.** greedy_select
- [x] **7.** build_vocab
- [x] **8.** encode_prompt
- [x] **9.** decode_tokens
- [x] **10.** embed_tokens
- [x] **11.** linear_projection
- [x] **12.** init_kv_cache
- [x] **13.** append_kv
- [x] **14.** causal_attention
- [x] **15.** model_prefill
- [x] **16.** model_decode_step
- [x] **17.** blocks_needed
- [x] **18.** init_block_allocator
- [x] **19.** allocate_block
- [x] **20.** free_block
- [x] **21.** append_to_paged_cache
- [x] **22.** gather_kv_from_blocks
- [x] **23.** paged_attention_step
- [x] **24.** free_sequence_blocks
- [x] **25.** kv_blocks_in_use
- [x] **26.** make_request
- [x] **27.** init_sequence_state
- [x] **28.** sequence_decode_step
- [x] **29.** is_sequence_done
- [x] **30.** generate_single_sequence
- [x] **31.** build_batch_step_input
- [x] **32.** batched_decode_step
- [x] **33.** static_batch_generate
- [x] **34.** has_free_capacity
- [x] **35.** continuous_batch_step
- [x] **36.** run_continuous_batching
- [x] **37.** priority_queue_push
- [x] **38.** priority_queue_pop
- [x] **39.** select_admissions
- [x] **40.** preempt_sequence
- [x] **41.** schedule_step
- [x] **42.** format_stream_chunk
- [x] **43.** submit_request
- [x] **44.** drive_until_complete
- [x] **45.** collect_request_output
- [x] **46.** build_completion_response
- [x] **47.** time_to_first_token
- [x] **48.** inter_token_latency
- [x] **49.** aggregate_throughput
- [x] **50.** latency_percentiles
- [x] **51.** run_throughput_latency_benchmark

## Results

```
[sampling] greedy=4, sampled=4, probs=[0.22270014 0.         0.         0.         0.77729986]
[vocab] size=31, bos=1, eos=2
[tokenize] prompt='hello world' ids=[1, 12, 9, 16, 16, 19, 4, 27, 19, 22, 16, 8] roundtrip='hello world'
[single] generated ids=[13, 7, 12, 13, 12, 7] text='ichihc'
[allocator] blocks=32, block_size=8, usage={'used': 0, 'free': 32, 'total': 32}
[server] submitted requests: ['req-0', 'req-1', 'req-2', 'req-3']
[server] req-0: tokens=[12, 24, 16, 24, 6] text='htltb'
[server] req-1: tokens=[2, 2, 18, 25, 2] text='nu'
[server] req-2: tokens=[14, 10, 10, 7, 22] text='jffcr'
[server] req-3: tokens=[10, 6, 6, 1, 9] text='fbbe'
[allocator] post-run usage={'used': 0, 'free': 32, 'total': 32}
[bench] wall=0.0120s report keys=['ttft', 'itl', 'throughput', 'percentiles', 'total_time']
  ttft: {'req-0': 0.0026743169999999816, 'req-1': 0.002696604999999963, 'req-2': 0.002717721999999978}
  itl: {'req-0': 6.994750000002825e-06, 'req-1': 6.069999999996911e-06, 'req-2': 5.958500000000089e-06}
  throughput: {'tokens_per_second': 5391.51375734597, 'requests_per_second': 1078.302751469194, 'total_tokens': 15, 'total_requests': 3}
  percentiles: {50.0: 0.002696604999999963, 90.0: 0.002713498599999975, 99.0: 0.0027172996599999777}
  total_time: 0.0028
```
