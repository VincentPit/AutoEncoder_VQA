import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from visual_embed.models import MAEEncoder, prepare_model

class MultiModalModel(nn.Module):
    def __init__(self, bert_model, vit_model, tokenizer, vocab_size, max_seq_length=1024):
        super(MultiModalModel, self).__init__()
        self.bert_model = bert_model
        self.vit_model = vit_model
        self.bert_proj = nn.Linear(768, 1024)  # Project BERT embeddings to match ViT size
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Initialize positional embedding with max_seq_length
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_length, 1024))

        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        decoder_layer = nn.TransformerDecoderLayer(d_model=1024, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.embedding = nn.Embedding(vocab_size, 1024)
        self.fc_out = nn.Linear(1024, vocab_size)

    def forward(self, text_input_ids, text_attention_mask, image_tensor, decoder_input_ids):
        # Get BERT embeddings
        text_outputs = self.bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeddings = text_outputs.last_hidden_state  # Shape: [batch_size, seq_len, 768]
        text_embeddings = self.bert_proj(text_embeddings)  # Shape: [batch_size, seq_len, 1024]

        # Get ViT embeddings
        image_embeddings = self.vit_model.forward(image_tensor)  # Shape: [batch_size, num_patches, 1024]

        # Combine text and image embeddings
        combined_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)  # Shape: [batch_size, total_seq_len, 1024]
        
        # Pad combined embeddings to max_seq_length
        if combined_embeddings.size(1) < self.max_seq_length:
            pad_size = self.max_seq_length - combined_embeddings.size(1)
            padding = torch.zeros(combined_embeddings.size(0), pad_size, 1024).to(combined_embeddings.device)
            combined_embeddings = torch.cat([combined_embeddings, padding], dim=1)
        else:
            combined_embeddings = combined_embeddings[:, :self.max_seq_length, :]

        # Add positional embeddings
        combined_embeddings += self.positional_embedding[:, :self.max_seq_length, :]

        # Pass through transformer encoder
        encoder_output = self.transformer_encoder(combined_embeddings)  # Shape: [batch_size, max_seq_length, 1024]

        # Prepare decoder input embeddings
        decoder_embeddings = self.embedding(decoder_input_ids)  # Shape: [batch_size, target_seq_len, 1024]

        # Pass through transformer decoder
        decoder_output = self.transformer_decoder(decoder_embeddings, encoder_output)  # Shape: [batch_size, target_seq_len, 1024]

        # Final classification layer
        output = self.fc_out(decoder_output)  # Shape: [batch_size, target_seq_len, vocab_size]

        return output

    def generate_answer(self, text_input_ids, text_attention_mask, image_tensor, max_length=50):
        decoder_input_ids = torch.tensor([[self.tokenizer.cls_token_id]]).to(text_input_ids.device)

        generated_answer = []

        for _ in range(max_length):
            output = self.forward(text_input_ids, text_attention_mask, image_tensor, decoder_input_ids)
            next_token_logits = output[:, -1, :]  # Get logits for the next token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # Greedy decoding
            generated_answer.append(next_token_id.item())

            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)

            if next_token_id.item() == self.tokenizer.sep_token_id:
                break

        generated_answer = self.tokenizer.decode(generated_answer, skip_special_tokens=True)
        return generated_answer

if __name__ == "__main__":
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(chkpt_dir='visual_embed/mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

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
