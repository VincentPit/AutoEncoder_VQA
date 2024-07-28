import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from tqdm import tqdm

# Import custom models and DataLoader
from cross_model import MultiModalModel
from dataloader import VQADataset 
from visual_embed.models import prepare_model

def train_model(model, dataloader, optimizer, criterion, tokenizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training Batches"):
        text_input_ids = batch['question'].to(device)
        text_attention_mask = (text_input_ids != tokenizer.pad_token_id).to(device)
        image_tensor = batch['image'].to(device)
        answer = batch['answer'].to(device)
        
        optimizer.zero_grad()
        
        output = model(text_input_ids, text_attention_mask, image_tensor, answer[:, :-1])
        
        # Shift targets to align with outputs
        target = answer[:, 1:].contiguous()
        
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize BERT and ViT models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(chkpt_dir='visual_embed/mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Create an instance of the multimodal model
    model = MultiModalModel(bert_model, vit_model, tokenizer, vocab_size)

    # Move model to device
    model.to(device)

    # Wrap model with DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Freeze BERT and ViT encoders
    for param in model.module.bert_model.parameters():
        param.requires_grad = False
    for param in model.module.vit_model.parameters():
        param.requires_grad = False

    # Load VQA dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    csv_file = 'dataset/data_train.csv'
    img_dir = 'dataset/images'
    dataset = VQADataset(csv_file=csv_file, img_dir=img_dir, tokenizer=tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_model(model, dataloader, optimizer, criterion, tokenizer, device)
        print(f"Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'multimodal_model.pth')
