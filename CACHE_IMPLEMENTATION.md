# Cache Handling for Incremental Inference

## Overview

This document provides an overview of the cache handling implementation for the Bayesian Self-Adaptive LLM project. The implementation enables efficient incremental inference for streaming interactions and autoregressive generation.

## Implementation Details

### 1. SVDMamba Cache Structure

The SVDMamba class implements caching with the following key components:

```python
# Internal cache variables
self._state_cache = None           # Stores hidden state information
self._cache_length = 0              # Tracks number of tokens processed
self._is_cache_initialized = False  # Cache initialization flag
```

The cache for Mamba state-space models contains:
- Last token's hidden state
- Last token's projected values
- Token index information

### 2. MambaformerLayer Cache Structure

The MambaformerLayer coordinates caching between the Mamba and attention components:

```python
# Attention KV cache
self.attn_kv_cache = None          # Stores key-value pairs for attention
self.cache_length = 0               # Tracks sequence length for attention

# Expert vector cache
self.last_expert = None             # Caches applied expert vector to avoid reapplication
```

The layer cache is a dictionary structure containing:
- `mamba_cache`: State cache for the Mamba block
- `attn_kv_cache`: Key-value cache for attention
- `cache_length`: Sequence length information

### 3. MambaFormer Model Cache Structure

The MambaFormer model coordinates caching across all layers:

```python
# Overall cache structure
{
    "layers": {
        "initial_mamba": { /* layer cache */ },
        "layer_0": { /* layer cache */ },
        "layer_1": { /* layer cache */ },
        ...
    }
}
```

### 4. Cache Flow During Inference

The cache flow during incremental inference follows these steps:

1. **First Token/Full Sequence Processing**:
   - Initialize cache for each layer
   - Store hidden states and attention key-values
   - Return cache dictionary if `return_cache=True`

2. **Subsequent Token Processing**:
   - Receive cache from previous step
   - Use cache to maintain context
   - Update cache with new token information
   - Return updated cache if `return_cache=True`

3. **Cache Reset**:
   - Called when context changes (e.g., expert vector updates)
   - Resets all layer caches to initial state

### 5. Handling Expert Vector Updates

When expert vectors are applied:
- Check if expert vector differs from last applied vector
- Apply expert vector to model weights
- Reset caches to ensure consistency with updated weights

## Usage Example

```python
# Process initial prompt
outputs = model(
    input_ids=initial_tokens,
    use_cache=True,
    expert_vector=expert_vector,
    return_dict=True
)
past_key_values = outputs["past_key_values"]

# Process next token incrementally
outputs = model(
    input_ids=next_token,
    use_cache=True,
    expert_vector=expert_vector,
    past_key_values=past_key_values,
    return_dict=True
)
past_key_values = outputs["past_key_values"]
```

## Design Considerations

1. **Memory Efficiency**: Caching only essential information to minimize memory footprint
2. **Computational Efficiency**: Avoiding redundant computations during autoregressive generation
3. **API Consistency**: Maintaining consistent interfaces with transformer models
4. **Expert Adaptation**: Ensuring cache consistency with expert vector changes

## Performance Implications

Properly implemented caching provides significant speedups:
- O(L) → O(1) complexity for processing each new token
- Particularly important for longer sequences
- Critical for real-time streaming applications

## Testing Approach

The cache implementation is tested with:
1. Verifying consistency between cached and non-cached outputs
2. Testing incremental token-by-token inference
3. Validating cache state after expert vector updates
4. Measuring performance improvements in autoregressive generation

## Future Optimizations

1. Quantization of cache states for memory efficiency
2. Optimizing cache memory layout for GPU efficiency
3. Implementing smart cache pruning for very long contexts
4. Adding sliding window caching for infinite-length streaming
