import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from tqdm import tqdm
from visual_embed.models import prepare_model
from mscoco_dataloader import COCOMatchDataset
from multimodal_matchup_model import MultiModalMatchupModel

# Paths to dataset directories and annotation files
image_dir = 'coco_dataset/train2017'
caption_file = 'coco_dataset/annotations/captions_train2017.json'

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the COCO dataset and DataLoader
coco_match_dataset = COCOMatchDataset(
    image_dir=image_dir,
    caption_file=caption_file,
    tokenizer=tokenizer,
    transform=transform
)
coco_match_dataloader = DataLoader(coco_match_dataset, batch_size=32, shuffle=True, num_workers=4)

# Load the pretrained models
bert_model = BertModel.from_pretrained('bert-base-uncased')
vit_model = prepare_model(
    chkpt_dir='visual_embed/mae_visualize_vit_large.pth',
    arch='mae_vit_large_patch16',
    only_encoder=True
)

# Freeze BERT and ViT models
for param in bert_model.parameters():
    param.requires_grad = False

for param in vit_model.parameters():
    param.requires_grad = False

# Initialize the multi-modal model
model = MultiModalMatchupModel(bert_model, vit_model, tokenizer.vocab_size)

# Freeze all layers except the encoder and decoder in MultiModalMatchupModel
for name, param in model.named_parameters():
    if 'encoder' not in name and 'decoder' not in name:
        param.requires_grad = False

# Check trainable parameters
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
print("Trainable Parameters:", trainable_params)

# Define optimizer with only the trainable parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device:", device)

# Training loop
num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(coco_match_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch'):
        images, input_ids, attention_masks = batch

        # Move tensors to the GPU
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        match_score = model(input_ids, attention_masks, images)

        # Dummy target (self-supervised learning), assuming match_score should be 1
        labels = torch.ones_like(match_score).to(device)
        
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(match_score, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print statistics at the end of the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(coco_match_dataloader):.4f}")

    # Save model checkpoint after each epoch
    checkpoint_path = f"checkpoints/model_checkpoint_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

# Final model save
final_model_path = "checkpoints/pretrained_matchup_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final trained model saved to {final_model_path}")

print("Training complete.")
