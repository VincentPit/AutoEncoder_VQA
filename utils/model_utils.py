"""
Base model classes and utilities for the AutoEncoder VQA project.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod


class BaseVQAModel(nn.Module, ABC):
    """Abstract base class for VQA models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        image_tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def generate_answer(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        image_tensor: torch.Tensor,
        **kwargs
    ) -> str:
        """Generate answer for given question and image."""
        pass
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
    
    def freeze_component(self, component_name: str) -> None:
        """Freeze parameters of a specific component."""
        if hasattr(self, component_name):
            component = getattr(self, component_name)
            for param in component.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Component '{component_name}' not found in model")
    
    def unfreeze_component(self, component_name: str) -> None:
        """Unfreeze parameters of a specific component."""
        if hasattr(self, component_name):
            component = getattr(self, component_name)
            for param in component.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Component '{component_name}' not found in model")


class MultiHeadCrossAttention(nn.Module):
    """Enhanced multi-head cross attention module."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        batch_first: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross attention.
        
        Args:
            query: Query tensor [batch_size, tgt_len, embed_dim]
            key: Key tensor [batch_size, src_len, embed_dim]
            value: Value tensor [batch_size, src_len, embed_dim]
            attn_mask: Attention mask
            key_padding_mask: Key padding mask
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)
        
        # Project to q, k, v
        Q = self.q_proj(query)  # [batch, tgt_len, embed_dim]
        K = self.k_proj(key)    # [batch, src_len, embed_dim]
        V = self.v_proj(value)  # [batch, src_len, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights.masked_fill_(attn_mask == 0, float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and put through final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        
        # Average attention weights across heads for visualization
        avg_attn_weights = attn_weights.mean(dim=1)
        
        return attn_output, avg_attn_weights


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight * x
        
        if self.bias is not None:
            x = x + self.bias
        
        return x


class TransformerBlock(nn.Module):
    """Enhanced transformer block with cross attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False
    ):
        super().__init__()
        
        self.norm_first = norm_first
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = LayerNorm(embed_dim, layer_norm_eps)
        
        # Cross attention (optional)
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim, num_heads, dropout=dropout
        )
        self.cross_attn_norm = LayerNorm(embed_dim, layer_norm_eps)
        
        # Feed forward
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout, activation)
        self.ffn_norm = LayerNorm(embed_dim, layer_norm_eps)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor
            memory: Memory tensor for cross attention
            tgt_mask: Target mask for self attention
            memory_mask: Memory mask for cross attention
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Tuple of (output, cross_attention_weights)
        """
        cross_attn_weights = None
        
        # Self attention
        if self.norm_first:
            x_norm = self.self_attn_norm(x)
            attn_output, _ = self.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask
            )
            x = x + self.dropout(attn_output)
        else:
            attn_output, _ = self.self_attn(
                x, x, x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask
            )
            x = self.self_attn_norm(x + self.dropout(attn_output))
        
        # Cross attention (if memory is provided)
        if memory is not None:
            if self.norm_first:
                x_norm = self.cross_attn_norm(x)
                cross_attn_output, cross_attn_weights = self.cross_attn(
                    x_norm, memory, memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask
                )
                x = x + self.dropout(cross_attn_output)
            else:
                cross_attn_output, cross_attn_weights = self.cross_attn(
                    x, memory, memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask
                )
                x = self.cross_attn_norm(x + self.dropout(cross_attn_output))
        
        # Feed forward
        if self.norm_first:
            x_norm = self.ffn_norm(x)
            ffn_output = self.ffn(x_norm)
            x = x + self.dropout(ffn_output)
        else:
            ffn_output = self.ffn(x)
            x = self.ffn_norm(x + self.dropout(ffn_output))
        
        return x, cross_attn_weights