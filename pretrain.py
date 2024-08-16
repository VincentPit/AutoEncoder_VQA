import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel 
from torchvision import transforms
from tqdm import tqdm  # Import tqdm for progress bars
from visual_embed.models import prepare_model
from mscoco_dataloader import COCOMatchDataset  # Import your custom DataLoader
from multimodal_matchup_model import MultiModalMatchupModel  # Import your model

# Ensure you have the appropriate files named `custom_dataloader.py` and `multimodal_matchup_model.py` where these classes are defined.

# Paths to dataset directories and annotation files
image_dir = 'coco_dataset/train2017'
caption_file = 'coco_dataset/annotations/captions_train2017.json'

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size

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

# Initialize the multi-modal model
model = MultiModalMatchupModel(bert_model, vit_model, vocab_size)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Wrap DataLoader with tqdm for progress bar
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
        
        # Assuming binary labels for match/no match (1/0)
        labels = torch.ones_like(match_score).to(device)  # Replace with actual labels when available
        
        # Compute loss
        loss = criterion(match_score, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print statistics at the end of the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(coco_match_dataloader):.4f}")

print("Training complete.")
