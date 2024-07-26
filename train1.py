import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from tqdm import tqdm

# Import custom models and DataLoader
from co_model import MultiModalModel
from dataloader import VQADataset 
from visual_embed.models import prepare_model

def train_model(rank, world_size, model, dataloader, optimizer, criterion, tokenizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training Batches", disable=(rank != 0)):
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    device = torch.device("cuda", rank)

    # Initialize BERT and ViT models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(chkpt_dir='visual_embed/mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Create an instance of the multimodal model
    model = MultiModalModel(bert_model, vit_model, tokenizer, vocab_size)
    
    # Move model to device
    model.to(device)

    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[rank])

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
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, sampler=sampler)

    # Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_model(rank, world_size, model, dataloader, optimizer, criterion, tokenizer, device)
        if rank == 0:
            print(f"Loss: {avg_loss:.4f}")

    # Save the trained model (only on rank 0 to avoid overwriting)
    if rank == 0:
        torch.save(model.module.state_dict(), 'multimodal_model_ddp.pth')

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
