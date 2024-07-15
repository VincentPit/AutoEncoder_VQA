import sys
import os
import torch
import torch.nn as nn
from question_embed.pretrained_bert import easy_bert
from visual_embed.models import MAEEncoder, prepare_model
from transformers import BertTokenizer, BertModel


class MultiModalModel(nn.Module):
    def __init__(self, bert_model, vit_model):
        super(MultiModalModel, self).__init__()
        self.bert_model = bert_model
        self.vit_model = vit_model
        self.bert_proj = nn.Linear(768, 1024)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(1024, 2)  # Example output layer for classification

    def forward(self, text_input_ids, text_attention_mask, image_tensor):
        # Get BERT embeddings
        text_outputs = self.bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeddings = text_outputs.last_hidden_state  # Shape: [batch_size, seq_len, 768]
        text_embeddings = self.bert_proj(text_embeddings)  # Shape: [batch_size, seq_len, 1024]

        # Get ViT embeddings
        image_embeddings = self.vit_model.forward(image_tensor)  # Shape: [batch_size, 197, 1024]

        # Concatenate along the sequence dimension
        combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)  # Shape: [batch_size, seq_len + 197, 1024]

        # Pass through transformer encoder layers
        transformer_output = self.transformer_encoder(combined_embeddings)  # Shape: [batch_size, seq_len + 197, 1024]

        # Final classification layer (example)
        output = self.fc(transformer_output[:, 0, :])  # Use [CLS] token representation for classification
        return output


if __name__ == "__main__":
    # Initialize BERT and ViT models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(chkpt_dir='visual_embed/mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True)

    # Create an instance of your multimodal model
    model = MultiModalModel(bert_model, vit_model)

    # Example input tensors (adjust according to your actual data)
    text_input_ids = torch.tensor([[101, 2023, 2003, 1037, 2518, 2003, 1037, 2062, 1010, 102]])
    text_attention_mask = torch.ones_like(text_input_ids)
    image_tensor = torch.randn(1, 3, 224, 224)  # Example image tensor

    # Ensure the model is in evaluation mode
    model.eval()

    # Perform a forward pass
    with torch.no_grad():
        output = model(text_input_ids, text_attention_mask, image_tensor)

    # Print the output
    print("Output:", output)
