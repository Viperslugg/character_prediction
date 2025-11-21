"""
Minimal decoder-only Transformer blocks in Flax/JAX, commented for learning.

The model mirrors a GPT-style (decoder-only) architecture:
- Token embeddings + learned positional embeddings
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

# Define separate class for Swish-Gated LU
class SwiGLU(nn.Module):
    """
    1. Have w1 and w2 produce the gate and data input respectively and separately
    2. Send w1 through the activation function SiLU --> gating mechanism
    3. Linear projection of input (value) --> data
    4. Merge by multiplication to project back to model dimension
    """
    hidden: int
    d_model: int

    @nn.compact
    def __call__(self, x):
        # The original PaLM paper proposes setting inner SwiGLU dimension
        # to 2/3 of standard FFN dimension
        hidden_dim = int(self.hidden * 2/3)
        
        # Added bias terms
        w1 = nn.Dense(hidden_dim, use_bias=True)(x)
        w2 = nn.Dense(hidden_dim, use_bias=True)(x)
        
        gate = nn.silu(w1)
        out = gate * w2
        # No final projection; consistent with MLP layer
        return out
    
# Define separate class for Ge-Gated LU
class GeGLU(nn.Module):
    """
    Formula: GELU(xW1 + b1) * (xW2 + b2)
    Flow: Similar to SwiGLU but using GeLU activation
    """
    hidden: int
    d_model: int

    @nn.compact
    def __call__(self, x):
        # Adjust hidden dimension (2/3 rule)
        hidden_dim = int(self.hidden * 2/3)

        # Two projections: one for gating, one for data
        w1 = nn.Dense(hidden_dim, use_bias=True)(x)
        w2 = nn.Dense(hidden_dim, use_bias=True)(x)

        # Apply GELU activation to gate
        gate = nn.gelu(w1)

        # Multiply and project back to d_model
        out = gate * w2
        # No Final projection; consistent with MLP layer
        return out


# Multi-Layer Perceptron block implements feed-forward network inside
# each Transformer block
class MLP(nn.Module):
        """
        Feed-forward sublayer of each Transformer block
        Linear(D -> 4D) -> activation -> Linear(4D -> D)
        Input shape:  (B, T, D)
        Output shape: (B, T, D)
        """

        d_model: int
        activation: str
        mlp_ratio: int
        dropout: float
        # For Flax dropout control
        deterministic: bool = True 

        @nn.compact
        def __call__(self, x):
            hidden = int(self.d_model * self.mlp_ratio)
            d_model = self.d_model
            
            if self.activation in ["GeGLU", "SwiGLU"]:
                if self.activation == "SwiGLU":
                    x = SwiGLU(hidden = hidden, d_model = d_model)(x)
                elif self.activation == "GeGLU":
                    x = GeGLU(hidden = hidden, d_model = d_model)(x)
            elif self.activation == "GeLU":
                 x = nn.Dense(hidden, use_bias=True)(x)
                 x = nn.gelu(x)

            elif self.activation == "SiLU":
                 x = nn.Dense(hidden, use_bias=True)(x)
                 x = nn.silu(x)
            else:
                 x = nn.Dense(hidden, use_bias=True)(x)
                 x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=self.deterministic)
            
            # Final projection
            x = nn.Dense(self.d_model, use_bias=True)(x)
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
    activation: str
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
             d_model=self.d_model, 
             mlp_ratio=self.mlp_ratio,
             dropout = self.dropout,
             deterministic = self.deterministic,
             activation = self.activation)(h)
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
    activation: str
    mlp_ratio: int
    dropout: float
    deterministic: bool = True

    def setup(self):
        # Token embedding table E with shape (V, D)
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)

        # (Learned, not sinusoidal or rotary) positional embeddings P with shape (max_len, D)
        # We'll slice P[:T] each forward pass and add to token embeddings.
        self.positional_embed = self.param(
            "positional_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)
        )

        # Stack of decoder blocks
        self.blocks = [DecoderBlock(d_model=self.d_model, 
                                    n_heads=self.n_heads, 
                                    mlp_ratio=self.mlp_ratio,
                                    dropout = self.dropout,
                                    deterministic=self.deterministic,
                                    activation=self.activation
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

        # Token + positional embeddings -> (B, T, D)
        x = self.tok_embed(idx) + self.positional_embed[:T]

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