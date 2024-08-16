import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from visual_embed.models import prepare_model

class MultiModalMatchupModel(nn.Module):
    def __init__(self, bert_model, vit_model, vocab_size):
        super(MultiModalMatchupModel, self).__init__()
        self.bert_model = bert_model
        self.vit_model = vit_model

        # Freeze ViT and BERT models
        for param in self.bert_model.parameters():
            param.requires_grad = False

        for param in self.vit_model.parameters():
            param.requires_grad = False

        # Define projection layers to a common dimension
        self.text_projection = nn.Linear(768, 1024)  # Project BERT embeddings to 1024
        self.image_projection = nn.Linear(1024, 1024)  # Project ViT embeddings to 1024

        # Similarity calculation
        self.similarity_score = nn.Linear(1024, 1)

    def forward(self, text_input_ids, text_attention_mask, image_tensor):
        # Get frozen BERT embeddings
        with torch.no_grad():
            text_embeddings = self.bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state
            # Use the [CLS] token for classification
            text_embeddings = text_embeddings[:, 0, :]
            text_embeddings = self.text_projection(text_embeddings)  # Project to 1024

        # Get frozen ViT embeddings
        with torch.no_grad():
            image_embeddings = self.vit_model.forward(image_tensor)  # [batch_size, num_patches, 1024]
            # Use the [CLS] token or aggregate patch embeddings (e.g., mean pooling)
            image_embeddings = image_embeddings.mean(dim=1)
            image_embeddings = self.image_projection(image_embeddings)  # Project to 1024

        # Compute similarity score
        similarity = self.compute_similarity(text_embeddings, image_embeddings)

        return similarity

    def compute_similarity(self, text_embeddings, image_embeddings):
        # Compute cosine similarity
        text_norm = torch.norm(text_embeddings, p=2, dim=-1, keepdim=True)
        image_norm = torch.norm(image_embeddings, p=2, dim=-1, keepdim=True)
        cosine_similarity = torch.mm(text_embeddings / text_norm, image_embeddings.T / image_norm)
        similarity_score = self.similarity_score(cosine_similarity).squeeze(-1)
        return similarity_score

    def compute_loss(self, text_embeddings, image_embeddings, target):
        # Project and calculate similarity
        text_embeddings = self.text_projection(text_embeddings)
        image_embeddings = self.image_projection(image_embeddings)
        similarity = self.compute_similarity(text_embeddings, image_embeddings)
        
        # Compute loss, e.g., using Contrastive Loss
        loss = nn.MSELoss()(similarity, target)
        return loss

if __name__ == "__main__":
    # Load models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(
        chkpt_dir='visual_embed/mae_visualize_vit_large.pth',
        arch='mae_vit_large_patch16',
        only_encoder=True
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    model = MultiModalMatchupModel(bert_model, vit_model, vocab_size)

    text_input_ids = torch.tensor([[101, 2023, 2003, 1037, 2518, 2003, 1037, 2062, 1010, 102]])
    text_attention_mask = torch.ones_like(text_input_ids)
    image_tensor = torch.randn(1, 3, 224, 224)

    model.eval()

    with torch.no_grad():
        similarity_score = model.forward(text_input_ids, text_attention_mask, image_tensor)
        print("Similarity Score:", similarity_score)
