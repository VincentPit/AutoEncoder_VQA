"""
Positional Embedding module for transformer models.

Implements sinusoidal positional embeddings as described in "Attention is All You Need".
"""

import torch
import torch.nn as nn
import math
from typing import Optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding layer.
    
    This implementation follows the original Transformer paper approach using
    sine and cosine functions of different frequencies.
    
    Args:
        max_seq_length (int): Maximum sequence length
        d_model (int): Model dimension (must be even)
        dropout (float): Dropout rate (default: 0.0)
    """
    
    def __init__(self, max_seq_length: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Create sinusoidal positional embeddings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term for sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it moves with the model but isn't a parameter
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_seq_length, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Positional embeddings of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_seq_length}"
            )
        
        # Extract positional embeddings for the sequence length
        pos_embedding = self.pe[:, :seq_len, :].to(x.device)
        
        if self.dropout is not None:
            pos_embedding = self.dropout(pos_embedding)
        
        return pos_embedding

def plot_positional_embeddings(
    positional_embedding: torch.Tensor, 
    title: str = "Positional Embeddings"
) -> None:
    """
    Plot positional embeddings as a heatmap.
    
    Args:
        positional_embedding (torch.Tensor): Positional embeddings to plot
        title (str): Plot title
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Cannot plot embeddings.")
        return
    
    # Convert to numpy and remove batch dimension if present
    pe = positional_embedding.squeeze(0).cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pe, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Embedding Value')
    plt.title(title)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position in Sequence')
    plt.tight_layout()
    plt.show()


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for self-attention.
    
    Args:
        seq_len (int): Sequence length
        device (torch.device, optional): Device to create tensor on
        
    Returns:
        torch.Tensor: Boolean mask of shape [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask


def create_padding_mask(
    input_ids: torch.Tensor, 
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Create padding mask for attention.
    
    Args:
        input_ids (torch.Tensor): Input token IDs
        pad_token_id (int): Padding token ID
        
    Returns:
        torch.Tensor: Boolean mask where True indicates padding positions
    """
    return input_ids == pad_token_id


def test_positional_embedding():
    """Test function for positional embeddings."""
    print("Testing Positional Embeddings...")
    print("=" * 40)
    
    # Test parameters
    max_seq_length = 128
    d_model = 512
    batch_size = 4
    seq_len = 64
    
    # Create positional embedding layer
    pos_emb = PositionalEmbedding(max_seq_length, d_model, dropout=0.1)
    
    print(f"Max sequence length: {max_seq_length}")
    print(f"Model dimension: {d_model}")
    print(f"Dropout rate: {pos_emb.dropout.p if pos_emb.dropout else 0.0}")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    
    # Get positional embeddings
    pos_embeddings = pos_emb(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Positional embeddings shape: {pos_embeddings.shape}")
    
    # Check properties
    print(f"Max positional embedding value: {pos_embeddings.max().item():.4f}")
    print(f"Min positional embedding value: {pos_embeddings.min().item():.4f}")
    print(f"Mean positional embedding value: {pos_embeddings.mean().item():.4f}")
    
    # Check if embeddings are deterministic (same for all batches)
    all_same = torch.allclose(pos_embeddings[0], pos_embeddings[1])
    print(f"Same embeddings across batch: {all_same}")
    
    # Test error handling
    try:
        long_input = torch.randn(1, max_seq_length + 10, d_model)
        pos_emb(long_input)
        print("ERROR: Should have raised ValueError for long sequence")
    except ValueError as e:
        print(f"✓ Correctly caught error for long sequence: {str(e)}")
    
    # Visualize if matplotlib is available
    if HAS_MATPLOTLIB:
        print("\nGenerating visualization...")
        plot_positional_embeddings(pos_embeddings[0:1], "Test Positional Embeddings")
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_positional_embedding()
