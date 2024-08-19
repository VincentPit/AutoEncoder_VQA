import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from visual_embed.models import MAEEncoder, prepare_model
from PIL import Image
from torchvision import transforms

class MultiModalModel(nn.Module):
    def __init__(self, bert_model, vit_model, tokenizer, vocab_size, max_seq_length=512, num_cross_attention_layers=4):
        super(MultiModalModel, self).__init__()
        self.bert_model = bert_model
        self.vit_model = vit_model
        self.bert_proj = nn.Linear(768, 1024)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.cross_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True) for _ in range(num_cross_attention_layers)]
        )

        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, 1024))  # Positional embeddings

        decoder_layer = nn.TransformerDecoderLayer(d_model=1024, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.embedding = nn.Embedding(vocab_size, 1024)
        self.fc_out = nn.Linear(1024, vocab_size)

    def forward(self, text_input_ids, text_attention_mask, image_tensor, decoder_input_ids):
        device = text_input_ids.device

        # Get BERT embeddings
        text_outputs = self.bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeddings = text_outputs.last_hidden_state  # Shape: [batch_size, seq_len, 768]
        text_embeddings = self.bert_proj(text_embeddings)  # Shape: [batch_size, seq_len, 1024]

        # Get ViT embeddings
        image_embeddings = self.vit_model(image_tensor.to(device))  # Shape: [batch_size, 197, 1024]

        # Align the dimensions
        if text_embeddings.size(1) > image_embeddings.size(1):
            diff = text_embeddings.size(1) - image_embeddings.size(1)
            padding = torch.zeros((image_embeddings.size(0), diff, image_embeddings.size(2)), device=device)
            image_embeddings = torch.cat([image_embeddings, padding], dim=1)
        elif image_embeddings.size(1) > text_embeddings.size(1):
            diff = image_embeddings.size(1) - text_embeddings.size(1)
            padding = torch.zeros((text_embeddings.size(0), diff, text_embeddings.size(2)), device=device)
            text_embeddings = torch.cat([text_embeddings, padding], dim=1)

        combined_embeddings = text_embeddings

        # Apply cross attention layers
        for cross_attention in self.cross_attention_layers:
            combined_embeddings, _ = cross_attention(text_embeddings, image_embeddings, combined_embeddings)

        # Add positional embeddings to the combined embeddings
        combined_embeddings += self.positional_encoding[:, :combined_embeddings.size(1), :]

        # Prepare decoder input embeddings
        decoder_embeddings = self.embedding(decoder_input_ids.to(device))  # Shape: [batch_size, target_seq_len, 1024]

        # Pass through transformer decoder layers
        decoder_output = self.transformer_decoder(decoder_embeddings, combined_embeddings)  # Shape: [batch_size, target_seq_len, 1024]

        # Final classification layer to predict the next word in the sequence
        output = self.fc_out(decoder_output)  # Shape: [batch_size, target_seq_len, vocab_size]

        return output

    def generate_answer(self, text_input_ids, text_attention_mask, image_tensor, max_length=50):
        device = text_input_ids.device

        # Initialize the decoder input with the [CLS] token
        decoder_input_ids = torch.tensor([[self.tokenizer.cls_token_id]], device=device)

        generated_answer = []

        for _ in range(max_length):
            output = self.forward(text_input_ids, text_attention_mask, image_tensor, decoder_input_ids)
            next_token_logits = output[:, -1, :]  # Get logits for the next token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # Greedy decoding
            generated_answer.append(next_token_id.item())

            # Append the predicted token to the decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)

            # Stop if the [SEP] token is generated
            if next_token_id.item() == self.tokenizer.sep_token_id:
                break

        # Decode the generated token IDs to text
        generated_answer = self.tokenizer.decode(generated_answer, skip_special_tokens=True)
        return generated_answer


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

if __name__ == "__main__":
    # Initialize BERT and ViT models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(chkpt_dir='visual_embed/mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Create an instance of your multimodal model
    model = MultiModalModel(bert_model, vit_model, tokenizer, vocab_size)

    # Example image paths and questions
    image_paths = [
        "train2014/COCO_train2014_000000458752.jpg",
        "train2014/COCO_train2014_000000262146.jpg"
    ]
    questions = [
        "What is this photo taken looking through?",
        "What position is this man playing?",
        "What color is the players shirt?",
        "Is this man a professional baseball player?",
        "What color is the snow?"
    ]

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for image_path in image_paths:
        for question in questions:
            # Preprocess inputs
            text_input_ids = tokenizer.encode(question, return_tensors="pt")
            text_attention_mask = torch.ones_like(text_input_ids)
            image_tensor = preprocess_image(image_path)

            # Move inputs to the same device as the model
            text_input_ids = text_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            image_tensor = image_tensor.to(device)

            # Ensure the model is in evaluation mode
            model.eval()

            # Get BERT embeddings
            with torch.no_grad():
                text_outputs = bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
                text_embeddings = text_outputs.last_hidden_state  # Shape: [batch_size, seq_len, 768]
                text_embeddings_proj = model.bert_proj(text_embeddings)  # Projected embeddings

            # Get ViT embeddings
            with torch.no_grad():
                image_embeddings = vit_model(image_tensor)  # Shape: [batch_size, 197, 1024]

            # Print BERT and ViT outputs
            print(f"Question: {question}")
            print(f"Image Path: {image_path}")
            print(f"BERT Embeddings (Projected): {text_embeddings_proj.squeeze().cpu().numpy()}")
            print(f"ViT Embeddings: {image_embeddings.squeeze().cpu().numpy()}")
            print()

            # Generate answer
            with torch.no_grad():
                answer = model.generate_answer(text_input_ids, text_attention_mask, image_tensor)

            # Print the generated answer
            print(f"Generated Answer: {answer}")
            print()