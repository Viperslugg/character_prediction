# functions.py
import numpy as np
import jax
import jax.numpy as jnp
import optax
import time
import matplotlib.pyplot as plt

# initialize the jax random key
key = jax.random.key(0)
seed = 42
rng = jax.random.PRNGKey(seed)

#####################################################################
# Loss function: Returns model performance metrics (cross-entropy loss and accuracy metrics)
@jax.jit
def loss_and_metrics(logits, targets):
    """
    Assumes `targets` contains only valid integer class ids in [0, V-1]

    Args: Model's predictions and correct tokens
      logits: (B_seq, B_tok, vocab_size)
      targets: (B_seq, B_tok) ground-truth class ids.

    Returns:
      loss: scalar average cross-entropy over all positions.
      metrics: a dict.
    """
    # Flatten batch/time dims into (N, V) and (N, ) so optax works
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)

    # Per-position cross-entropy, then mean over all positions
    per_pos = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets)
    loss = per_pos.mean()

    # prediction, comparison with actual
    preds = jnp.argmax(logits, axis=-1)
    is_match = preds == targets

    # Accuracy over all tokens
    acc_all = jnp.mean(is_match.astype(jnp.float32))

    # Accuracy over only last token of each sequence
    acc_last = jnp.mean(is_match.astype(jnp.float32)[:,-1])

    return loss, {"loss": loss, "acc": acc_all, "acc_last": acc_last}
####################################################################

###########################################################################
# Batch creation
def get_batch(text_int, B_seq, B_tok):
    """
    Args:
      text_int: 1D array of token ids.
      B_seq: batch size (number of sequences per batch).
      B_tok: sequence length (number of tokens per sequence in batch).

    Returns:
      x, y with shapes: (B_seq, B_tok) int array input tokens.
      x: inputs
      y: targets
    """
    # choose random starting index for each sequence in the batch
    ix = np.random.randint(0, len(text_int) - B_tok, size=B_seq)
    x = np.stack([text_int[i:i+B_tok] for i in ix])
    y = np.stack([text_int[i+1:i+B_tok+1] for i in ix])
    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)
