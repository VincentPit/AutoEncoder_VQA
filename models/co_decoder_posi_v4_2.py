import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from visual_embed.models import MAEEncoder, prepare_model
from positional_embedding import PositionalEmbedding
import torch.nn.functional as F


class MultiModalModel(nn.Module):
    def __init__(self, bert_model, vit_model, tokenizer, vocab_size, max_seq_length=1024, dropout_rate=0.1):
        super(MultiModalModel, self).__init__()
        self.bert_model = bert_model
        self.vit_model = vit_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Positional embedding
        self.positional_embedding = PositionalEmbedding(max_seq_length, 768)  # Text embedding size is 768

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Cross-attention layer (image-to-text attention)
        self.cross_attention_layer = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)

        # Encoder for ViT output with bi-self-attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True)
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Linear projection to align encoder output with text embedding size
        self.vit_projection = nn.Linear(1024, 768)

        # Additional cross-attention layer (text-to-ViT attention before decoder)
        self.pre_decoder_cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Main transformer decoder with causal self-attention and cross-attention to ViT output
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Embedding and final classification layer
        self.embedding = nn.Embedding(vocab_size, 768)
        self.fc_out = nn.Linear(768, vocab_size)

    def forward(self, text_input_ids, text_attention_mask, image_tensor, decoder_input_ids):
        # Get BERT embeddings
        text_input_ids = text_input_ids.long()
        
        text_outputs = self.bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeddings = text_outputs.last_hidden_state  # Shape: [batch_size, seq_len, 768]

        # Apply dropout to BERT embeddings
        #text_embeddings = self.dropout(text_embeddings)

        # Get ViT embeddings
        image_embeddings = self.vit_model.forward(image_tensor)  # Shape: [batch_size, num_patches, 1024]

        # Apply cross-attention between image and text embeddings (image-to-text attention)
        text_embeddings_proj = torch.cat([text_embeddings, torch.zeros(text_embeddings.size(0), text_embeddings.size(1), 256).to(text_embeddings.device)], dim=-1)
        cross_attention_output, _ = self.cross_attention_layer(query=image_embeddings, key=text_embeddings_proj, value=text_embeddings_proj)

        # Apply bi-self-attention to the cross-attention output
        vit_encoder_output = self.vit_encoder(cross_attention_output)  # Shape: [batch_size, num_patches, 1024]

        # Project encoder output to match text embedding size
        vit_output_aligned = self.vit_projection(vit_encoder_output)  # Shape: [batch_size, num_patches, 768]

        # Add positional embeddings to the output of bi-self-attention
        vit_output_aligned += self.positional_embedding(vit_output_aligned)

        # Apply dropout to the projected ViT output
        vit_output_aligned = self.dropout(vit_output_aligned)

        # Apply cross-attention between text embeddings and ViT encoder output before the decoder
        pre_decoder_attention_output, _ = self.pre_decoder_cross_attention(query=text_embeddings, key=vit_output_aligned, value=vit_output_aligned)

        # Prepare decoder input embeddings
        decoder_input_ids= decoder_input_ids.long()
        decoder_embeddings = self.embedding(decoder_input_ids)  # Shape: [batch_size, target_seq_len, 768]

        # Apply dropout to decoder embeddings
        decoder_embeddings = self.dropout(decoder_embeddings)

        # Pass through the decoder with causal self-attention and cross-attention to the pre-decoder cross-attention output
        decoder_output = self.transformer_decoder(decoder_embeddings, pre_decoder_attention_output)  # Shape: [batch_size, target_seq_len, 768]

        # Apply dropout to decoder output
        decoder_output = self.dropout(decoder_output)

        # Final classification layer
        output = self.fc_out(decoder_output)  # Shape: [batch_size, target_seq_len, vocab_size]

        return output

    def generate_answer(self, text_input_ids, text_attention_mask, image_tensor, beam_size=3, max_length=50):
        decoder_input_ids = torch.tensor([[self.tokenizer.cls_token_id]]).to(text_input_ids.device)
        beam = [(decoder_input_ids, 0.0)]  # List of tuples (sequence, score)

        for _ in range(max_length):
            candidates = []
            for seq, score in beam:
                output = self.forward(text_input_ids, text_attention_mask, image_tensor, seq)
                next_token_logits = output[:, -1, :]  # Get logits for the next token
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)  # Convert logits to log-probabilities
                
                topk_probs, topk_ids = torch.topk(next_token_probs, beam_size, dim=-1)  # Get top k candidates
                for i in range(beam_size):
                    candidate_seq = torch.cat([seq, topk_ids[:, i].unsqueeze(1)], dim=1)
                    candidate_score = score + topk_probs[0, i].item()
                    candidates.append((candidate_seq, candidate_score))

            # Sort candidates by score and select top `beam_size` sequences
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            # Stop if all sequences end with the SEP token
            if all(seq[0][-1].item() == self.tokenizer.sep_token_id for seq, _ in beam):
                break

        # Select the sequence with the highest score
        best_seq = beam[0][0]
        generated_answer = self.tokenizer.decode(best_seq.squeeze(), skip_special_tokens=True)
        return generated_answer

if __name__ == "__main__":
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(chkpt_dir='visual_embed/mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    print("vocab size:", vocab_size)
    model = MultiModalModel(bert_model, vit_model, tokenizer, vocab_size)

    text_input_ids = torch.tensor([[101, 2023, 2003, 1037, 2518, 2003, 1037, 2062, 1010, 102]])
    text_attention_mask = torch.ones_like(text_input_ids)
    image_tensor = torch.randn(1, 3, 224, 224)  # Example image tensor
    decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]])

    model.eval()

    with torch.no_grad():
        output = model.forward(text_input_ids, text_attention_mask, image_tensor, decoder_input_ids)
        print("Model output shape:", output.shape)

        answer = model.generate_answer(text_input_ids, text_attention_mask, image_tensor)
        print("Generated Answer:", answer)