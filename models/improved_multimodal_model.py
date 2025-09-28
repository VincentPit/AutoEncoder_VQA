"""
Improved MultiModal VQA Model with enhanced architecture.

This module contains the main VQA model that combines BERT text encoding
with Vision Transformer image encoding and uses cross-attention mechanisms
for multimodal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from transformers import BertModel, BertTokenizer

from visual_embed.models import MAEEncoder, prepare_model
from models.positional_embedding import PositionalEmbedding
from utils.model_utils import BaseVQAModel, TransformerBlock


class ImprovedMultiModalModel(BaseVQAModel):
    """
    Improved multimodal VQA model with enhanced cross-attention mechanisms.
    
    This model combines:
    - BERT for text understanding
    - Vision Transformer with MAE for image understanding
    - Cross-attention layers for multimodal fusion
    - Transformer decoder for answer generation
    """
    
    def __init__(
        self,
        bert_model: BertModel,
        vit_model: nn.Module,
        tokenizer: BertTokenizer,
        config: Dict[str, Any]
    ):
        """
        Initialize the improved multimodal model.
        
        Args:
            bert_model: Pre-trained BERT model
            vit_model: Pre-trained Vision Transformer model
            tokenizer: BERT tokenizer
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        self.bert_model = bert_model
        self.vit_model = vit_model
        self.tokenizer = tokenizer
        
        # Model configuration
        self.max_seq_length = config.get('max_seq_length', 512)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.num_attention_heads = config.get('num_attention_heads', 8)
        self.hidden_size = config.get('hidden_size', 768)
        self.vocab_size = config.get('vocab_size', tokenizer.vocab_size)
        
        # Positional embeddings
        self.text_pos_embedding = PositionalEmbedding(
            self.max_seq_length, self.hidden_size
        )
        
        # Visual feature projection
        self.vit_projection = nn.Sequential(
            nn.Linear(1024, self.hidden_size),  # ViT output is 1024-dim
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout_rate)
        )
        
        # Cross-attention layers for multimodal fusion
        self.cross_attention_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.hidden_size,
                num_heads=self.num_attention_heads,
                ffn_dim=self.hidden_size * 4,
                dropout=self.dropout_rate,
                activation="gelu",
                norm_first=True
            )
            for _ in range(config.get('cross_attention_layers', 4))
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.hidden_size,
                num_heads=self.num_attention_heads,
                ffn_dim=self.hidden_size * 4,
                dropout=self.dropout_rate,
                activation="gelu",
                norm_first=True
            )
            for _ in range(config.get('decoder_layers', 6))
        ])
        
        # Answer generation components
        self.answer_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.answer_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.vocab_size)
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(self.hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode text using BERT.
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Text embeddings [batch_size, seq_len, hidden_size]
        """
        text_outputs = self.bert_model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask
        )
        text_embeddings = text_outputs.last_hidden_state
        
        # Add positional embeddings
        pos_embeddings = self.text_pos_embedding(text_embeddings)
        text_embeddings = text_embeddings + pos_embeddings
        
        return self.dropout(text_embeddings)
    
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode image using Vision Transformer.
        
        Args:
            image_tensor: Input image [batch_size, 3, 224, 224]
            
        Returns:
            Image embeddings [batch_size, num_patches, hidden_size]
        """
        with torch.no_grad():
            vit_features = self.vit_model(image_tensor)  # [batch_size, num_patches, 1024]
        
        # Project to hidden size
        image_embeddings = self.vit_projection(vit_features)
        
        return image_embeddings
    
    def cross_modal_fusion(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform cross-modal fusion using cross-attention.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_len, hidden_size]
            image_embeddings: Image embeddings [batch_size, img_len, hidden_size]
            text_attention_mask: Text attention mask [batch_size, text_len]
            
        Returns:
            Tuple of (fused_text_features, fused_image_features)
        """
        # Create attention masks
        text_key_padding_mask = (text_attention_mask == 0)
        
        fused_text = text_embeddings
        fused_image = image_embeddings
        
        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            # Text attending to image
            fused_text, _ = layer(
                fused_text,
                memory=fused_image,
                tgt_key_padding_mask=text_key_padding_mask
            )
            
            # Image attending to text
            fused_image, _ = layer(
                fused_image,
                memory=fused_text,
                memory_key_padding_mask=text_key_padding_mask
            )
        
        return fused_text, fused_image
    
    def generate_answer_logits(
        self,
        fused_features: torch.Tensor,
        decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate answer logits using the decoder.
        
        Args:
            fused_features: Fused multimodal features [batch_size, seq_len, hidden_size]
            decoder_input_ids: Decoder input IDs [batch_size, target_len]
            
        Returns:
            Answer logits [batch_size, target_len, vocab_size]
        """
        # Embed decoder inputs
        decoder_embeddings = self.answer_embedding(decoder_input_ids)
        
        # Add positional embeddings
        pos_embeddings = self.text_pos_embedding(decoder_embeddings)
        decoder_embeddings = decoder_embeddings + pos_embeddings
        decoder_embeddings = self.dropout(decoder_embeddings)
        
        # Create causal mask for decoder
        target_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(target_len, target_len, dtype=torch.bool, device=decoder_input_ids.device),
            diagonal=1
        )
        
        # Apply decoder layers
        decoder_output = decoder_embeddings
        for layer in self.decoder_layers:
            decoder_output, _ = layer(
                decoder_output,
                memory=fused_features,
                tgt_mask=causal_mask
            )
        
        # Final normalization and projection
        decoder_output = self.final_norm(decoder_output)
        logits = self.answer_projection(decoder_output)
        
        return logits
    
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        image_tensor: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            text_input_ids: Text token IDs [batch_size, text_len]
            text_attention_mask: Text attention mask [batch_size, text_len]
            image_tensor: Image tensor [batch_size, 3, 224, 224]
            decoder_input_ids: Decoder input IDs [batch_size, target_len]
            
        Returns:
            Answer logits [batch_size, target_len, vocab_size]
        """
        # Encode text and image
        text_embeddings = self.encode_text(text_input_ids, text_attention_mask)
        image_embeddings = self.encode_image(image_tensor)
        
        # Cross-modal fusion
        fused_text, fused_image = self.cross_modal_fusion(
            text_embeddings, image_embeddings, text_attention_mask
        )
        
        # Concatenate fused features
        fused_features = torch.cat([fused_text, fused_image], dim=1)
        
        # Generate answer logits
        logits = self.generate_answer_logits(fused_features, decoder_input_ids)
        
        return logits
    
    def generate_answer(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        image_tensor: torch.Tensor,
        max_length: int = 50,
        beam_size: int = 5,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate answer using beam search.
        
        Args:
            text_input_ids: Text token IDs [batch_size, text_len]
            text_attention_mask: Text attention mask [batch_size, text_len]
            image_tensor: Image tensor [batch_size, 3, 224, 224]
            max_length: Maximum answer length
            beam_size: Beam search width
            temperature: Sampling temperature
            
        Returns:
            Generated answer as string
        """
        self.eval()
        
        with torch.no_grad():
            # Encode inputs
            text_embeddings = self.encode_text(text_input_ids, text_attention_mask)
            image_embeddings = self.encode_image(image_tensor)
            
            # Cross-modal fusion
            fused_text, fused_image = self.cross_modal_fusion(
                text_embeddings, image_embeddings, text_attention_mask
            )
            
            # Concatenate fused features
            fused_features = torch.cat([fused_text, fused_image], dim=1)
            
            # Initialize beam search
            batch_size = text_input_ids.size(0)
            device = text_input_ids.device
            
            # Start with CLS token
            decoder_input_ids = torch.full(
                (batch_size, 1), 
                self.tokenizer.cls_token_id, 
                dtype=torch.long, 
                device=device
            )
            
            # Beam search
            finished_sequences = []
            beam = [(decoder_input_ids, 0.0)]
            
            for step in range(max_length):
                candidates = []
                
                for seq, score in beam:
                    if seq[0, -1].item() == self.tokenizer.sep_token_id:
                        finished_sequences.append((seq, score))
                        continue
                    
                    # Generate logits for current sequence
                    logits = self.generate_answer_logits(fused_features, seq)
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Get top-k candidates
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)
                    
                    for i in range(beam_size):
                        token_id = top_indices[:, i:i+1]
                        token_score = top_log_probs[:, i].item()
                        
                        new_seq = torch.cat([seq, token_id], dim=1)
                        new_score = score + token_score
                        candidates.append((new_seq, new_score))
                
                # Select top candidates
                candidates.extend(finished_sequences)
                candidates.sort(key=lambda x: x[1], reverse=True)
                beam = candidates[:beam_size]
                
                # Early stopping if all sequences are finished
                if len(finished_sequences) >= beam_size:
                    break
            
            # Select best sequence
            if finished_sequences:
                best_seq = max(finished_sequences, key=lambda x: x[1])[0]
            else:
                best_seq = beam[0][0]
            
            # Decode to string
            answer = self.tokenizer.decode(
                best_seq.squeeze(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return answer.strip()
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for answer generation.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            target_ids: Target token IDs [batch_size, seq_len]
            target_mask: Target mask [batch_size, seq_len]
            
        Returns:
            Loss tensor
        """
        # Flatten logits and targets
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.tokenizer.pad_token_id,
            reduction='none'
        )
        
        # Apply mask if provided
        if target_mask is not None:
            mask_flat = target_mask.view(-1)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum()
        else:
            loss = loss.mean()
        
        return loss