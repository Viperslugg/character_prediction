"""
Minimal decoder-only Transformer blocks in Flax/JAX, commented for learning.

The model mirrors a GPT-style (decoder-only) architecture:
- Token embeddings + (learned/sinusoidal/rotary) positional embeddings
- Stack of Pre-LayerNorm decoder blocks with causal self-attention
- Final LayerNorm
- Weight tying between input embeddings and output logits projection

Tensor shape conventions used below:
- B: batch size
- T: sequence length (time/positions)
- D: hidden size / embedding dimension (d_model)
- V: vocabulary size
"""

# import Linen API from Flax; defines neural network modules
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

# Multi-Layer Perceptron block implements feed-forward network inside
# each Transformer block
class MLP(nn.Module):
        """
        Feed-forward sublayer of each Transformer block
        mlp_ratio affects the feedforward dimension; currently it is 4
        Linear(D -> 4D) -> GELU activation -> Linear(4D -> D)
        Input shape:  (B, T, D)
        Output shape: (B, T, D)
        """

        d_model: int
        mlp_ratio: int
        dropout: float
        # For reproducibility
        deterministic: bool = True

        @nn.compact
        def __call__(self, x):
            hidden = int(self.d_model * self.mlp_ratio)
            x = nn.Dense(hidden)(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=self.deterministic)
            x = nn.Dense(self.d_model)(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=self.deterministic)
            return x

class DecoderBlock(nn.Module):
    """
    Each block represents one layer in the transformer
    Args:
      d_model: Hidden size D.
      n_heads: Number of attention heads.

    Input/Output shape: (B, T, D)
    """

    d_model: int
    n_heads: int
    # feedforward dimension (MLP ratio) can be changed
    # A common practice is to set it to 4
    mlp_ratio: int
    dropout: float
    deterministic: bool = True 

    # Added dropout inside both attention and MLP sublayers
    @nn.compact
    def __call__(self, x, *, mask=None):
        # Self-attention via causal mask
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            use_bias=False,
            dropout_rate = self.dropout,
            deterministic = self.deterministic
        )(h, mask=mask)
        x = x + h  # residual connection

        # MLP sublayer: Pre-LayerNorm -> MLP -> Residual add
        h = nn.LayerNorm()(x)
        h = MLP(
             self.d_model, 
             mlp_ratio=self.mlp_ratio,
             dropout = self.dropout,
             deterministic = self.deterministic)(h)
        x = x + h  # residual connection
        return x

class DecoderOnlyTransformer(nn.Module):
    """GPT-style decoder-only Transformer for language modeling.

    Args:
      vocab_size: no of unique tokens (characters)
      d_model: embedding dimension
      n_layers: Number of decoder blocks.
      n_heads: Attention heads per block.
      max_len: Maximum sequence length for positional embeddings.
      mlp_ratio: Ratio of hidden layer to embedding dimension
      dropout: Regularization to prevent overfitting
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int
    dropout: float
    deterministic: bool = True

    def sinusoidal_pos(self):
         d_model = self.d_model
         max_len = self.max_len
         N = 10000
         # Generate a column vector of positions
         numerator = jnp.arange(max_len)[:, None]
         denom = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(N)/d_model))

         pos = numerator * denom
         pos_embeddings = jnp.zeros((max_len, d_model))
         # sin(pos / 10_000^(2i/d))
         pos_embeddings = pos_embeddings.at[:, 0::2].set(jnp.sin(pos))
         # Use cosine instead
         pos_embeddings = pos_embeddings.at[:, 1::2].set(jnp.cos(pos))
         return pos_embeddings    # Shape: (max_len, d_model)

    def setup(self):
        # Token embedding table E with shape (V, D)
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)

        """
        # Learned positional embedding P with shape (max_len, D)
        self.positional_embed = self.param(
            "positional_embed",
            # same initialization style used in original transformer
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)      # shape of param tensor
        )
        # Token + positional embeddings -> (B, T, D)
        # Slice P[:T] each forward pass and add to token embeddings
        """

        # Stack of decoder blocks
        self.blocks = [DecoderBlock(d_model=self.d_model, 
                                    n_heads=self.n_heads, 
                                    mlp_ratio=self.mlp_ratio,
                                    dropout = self.dropout,
                                    deterministic=self.deterministic
                                    ) for _ in range(self.n_layers)]

        # Final LayerNorm
        self.layerNorm_final = nn.LayerNorm()

        # Output projection (to vocab logits)
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, idx):
        """Forward pass (causal-only).

        Args:
          idx: Token ids of shape (B, T), dtype int32/int64.

        Returns:
          logits: (B, T, V) unnormalized vocabulary scores for next-token prediction.
        """
        B, T = idx.shape

        pos_embed = self.sinusoidal_pos()[:T]
        x = self.tok_embed(idx) + pos_embed
        # x = self.tok_embed(idx) + self.positional_embed[:T]

        # Build causal attention mask
        # lower-triangular mask so pos t can only react to <= t (no "future" tokens)
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        mask = causal

        # Pass through each decoder block performing self-attention
        # + MLP + residual connections
        for blk in self.blocks:
            x = blk(x, mask=mask)

        # Final LayerNorm
        x = self.layerNorm_final(x)

        # Output projection to logits over V tokens.
        logits = self.project_to_vocab(x)
        
        return logits