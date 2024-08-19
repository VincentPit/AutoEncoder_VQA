import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from visual_embed.models import prepare_model
import torch.nn.functional as F

class MultiModalMatchupModel(nn.Module):
    def __init__(self, bert_model, vit_model, vocab_size, hidden_dim=1024, num_encoder_layers=4, num_decoder_layers=4):
        super(MultiModalMatchupModel, self).__init__()
        self.bert_model = bert_model
        self.vit_model = vit_model

        # Freeze ViT and BERT models
        for param in self.bert_model.parameters():
            param.requires_grad = False

        for param in self.vit_model.parameters():
            param.requires_grad = False

        # Define projection layers
        self.text_projection = nn.Linear(768, hidden_dim)
        self.image_projection = nn.Linear(1024, hidden_dim)  # Added image projection

        # Define the bidirectional transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Define the transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Define the similarity score layer
        #self.similarity_score = nn.Linear(hidden_dim, 1)

    def forward(self, text_input_ids, text_attention_mask, image_tensor):
        # Get frozen BERT embeddings
        with torch.no_grad():
            text_embeddings = self.bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state
            text_embeddings = self.text_projection(text_embeddings)

        # Get frozen ViT embeddings
        with torch.no_grad():
            image_embeddings = self.vit_model(image_tensor)  # [batch_size, num_patches, 1024]
            image_embeddings = self.image_projection(image_embeddings)  # [batch_size, num_patches, hidden_dim]

        text_embeddings = text_embeddings.mean(dim=1)  # Aggregate text embeddings to [batch_size, hidden_dim]
        text_embeddings = text_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        image_embeddings = image_embeddings.mean(dim=1)  # Aggregate image embeddings to [batch_size, hidden_dim]
        image_embeddings = image_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Normalize embeddings
        text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=-1)

        # Encode image embeddings using the transformer encoder
        memory = self.transformer_encoder(image_embeddings)

        # Pass through the transformer decoder
        decoder_output = self.transformer_decoder(text_embeddings, memory)
        decoder_output = decoder_output.mean(dim=1)  # Aggregate decoder output to [batch_size, hidden_dim]

        # Debugging shapes
        #print("decoder_output:", decoder_output.shape)
        #print("image_embeddings:", image_embeddings.squeeze(1).shape)

        # Compute similarity score
        similarity = self.compute_similarity(decoder_output, image_embeddings.squeeze(1))

        return similarity

    def compute_similarity(self, text_embeddings, image_embeddings):
        # Ensure the shapes are correct: [batch_size, hidden_dim]
        text_embeddings = text_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        image_embeddings = image_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Compute cosine similarity
        text_norm = torch.norm(text_embeddings, p=2, dim=-1, keepdim=True)  # [batch_size, 1, 1]
        image_norm = torch.norm(image_embeddings, p=2, dim=-1, keepdim=True)  # [batch_size, 1, 1]
        text_embeddings = text_embeddings / text_norm  # [batch_size, 1, hidden_dim]
        image_embeddings = image_embeddings / image_norm  # [batch_size, 1, hidden_dim]
        
        cosine_similarity = torch.bmm(image_embeddings, text_embeddings.permute(0, 2, 1))  # [batch_size, 1, 1]
        
        
        #print( cosine_similarity, " cosine_similarity shape:",  cosine_similarity.shape)
        
        #similarity_score = self.similarity_score(cosine_similarity)  # [batch_size, 1]

        return cosine_similarity[0][0][0]

    def compute_loss(self, text_embeddings, image_embeddings, target):
        text_embeddings = self.text_projection(text_embeddings)
        image_embeddings = self.image_projection(image_embeddings)
        similarity = self.compute_similarity(text_embeddings, image_embeddings)
        
        # Compute loss using Mean Squared Error Loss
        loss = nn.MSELoss()(similarity, target)
        return loss

def test_forward_pass():
    # Load models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(
        chkpt_dir='visual_embed/mae_visualize_vit_large.pth',
        arch='mae_vit_large_patch16',
        only_encoder=True
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Initialize the multi-modal model
    model = MultiModalMatchupModel(bert_model, vit_model, vocab_size)

    # Dummy input data
    text_input_ids = torch.tensor([[101, 2023, 2003, 1037, 2518, 2003, 1037, 2062, 1010, 102]])  # Example token IDs
    text_attention_mask = torch.ones_like(text_input_ids)  # Example attention mask
    image_tensor = torch.randn(1, 3, 224, 224)  # Example image tensor

    # Move model to evaluation mode
    model.eval()

    # Perform a forward pass with no gradient calculation
    with torch.no_grad():
        similarity_score = model(text_input_ids, text_attention_mask, image_tensor)
        print("Similarity Score:", similarity_score)

if __name__ == "__main__":
    test_forward_pass()
