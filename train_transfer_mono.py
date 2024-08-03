import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import os
import random

# Import custom models and DataLoader
from transfer_cross_attention_model import MultiModalModel
from dataloader import VQADataset
from visual_embed.models import prepare_model

def train_model(model, dataloader, optimizer, criterion, tokenizer, device, clip_value=1.0):
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_answer(model, tokenizer, image_path, question, device, max_length=50):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    inputs = tokenizer(question, return_tensors="pt", padding='max_length', max_length=512, truncation=True)
    text_input_ids = inputs['input_ids'].to(device)
    text_attention_mask = inputs['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        answer = model.generate_answer(text_input_ids, text_attention_mask, image_tensor, max_length=max_length)

    return answer

def evaluate_initial_model(model, eval_dataloader, tokenizer, img_dir, device):
    model.eval()
    print("Initial Evaluation on some QA pairs before training:")
    
    for i, eval_example in enumerate(eval_dataloader):
        image_path = os.path.join(img_dir, f"{eval_example['img_id'][0]}.png")
        question = eval_example['question_text'][0]
        
        answer = generate_answer(model, tokenizer, image_path, question, device)
        
        print(f"Evaluation example {i+1} - Question: {question}")
        print(f"Generated Answer: {answer}")
        
        # Evaluate on first 5 examples
        if i >= 4:
            break

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

    # Freeze BERT and ViT encoders
    for param in model.bert_model.parameters():
        param.requires_grad = False
    for param in model.vit_model.parameters():
        param.requires_grad = False

    # Load VQA dataset for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    csv_file = 'dataset/data_train.csv'
    img_dir = 'dataset/images'
    train_dataset = VQADataset(csv_file=csv_file, img_dir=img_dir, tokenizer=tokenizer, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Load VQA dataset for evaluation
    eval_csv_file = 'dataset/data_eval.csv'
    eval_dataset = VQADataset(csv_file=eval_csv_file, img_dir=img_dir, tokenizer=tokenizer, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Initial evaluation before training
    evaluate_initial_model(model, eval_dataloader, tokenizer, img_dir, device)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_model(model, train_dataloader, optimizer, criterion, tokenizer, device)
        scheduler.step()
        print(f"Loss: {avg_loss:.4f}")

        # Generate and print answer for five random examples in the eval dataset
        eval_indices = random.sample(range(len(eval_dataloader.dataset)), 5)
        for idx in eval_indices:
            eval_example = eval_dataloader.dataset[idx]
            image_path = os.path.join(img_dir, f"{eval_example['img_id']}.png")
            question = eval_example['question_text']
        
            answer = generate_answer(model, tokenizer, image_path, question, device)
        
            print(f"Evaluation example - Question: {question}")
            print(f"Generated Answer: {answer}")

    # Save the trained model
    torch.save(model.state_dict(), 'transfer_cross_attention10.pth')
