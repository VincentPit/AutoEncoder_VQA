import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler  # For mixed precision
from PIL import Image
import os
import random
import numpy as np

# Import custom models and DataLoader
from co_model import MultiModalModel
from coco_dataloader import CocoVQADataset
from visual_embed.models import prepare_model

def train_model(model, dataloader, optimizer, criterion, tokenizer, device, scaler, clip_value=1.0, epsilon=1e-8, accumulation_steps=2):
    model.train()
    total_loss = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Training Batches")):
        if not validate_batch(batch):
            continue

        text_input_ids = batch['question'].to(device)
        text_attention_mask = (text_input_ids != tokenizer.pad_token_id).to(device)
        image_tensor = batch['image'].to(device)
        answer = batch['answer'].to(device)

        with autocast():  # Mixed Precision Training
            output = model(text_input_ids, text_attention_mask, image_tensor, answer[:, :-1])
            target = answer[:, 1:].contiguous()

            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss = loss + epsilon

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_batch(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):  # Ensure the value is a tensor before checking
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"Warning: {key} contains nan or inf values.")
                return False
    return True

def showcase_predictions(model, dataloader, tokenizer, device, num_samples=5):
    model.eval()
    print("\nShowcasing Model Predictions:")
    
    for i, batch in enumerate(dataloader):
        if i >= 1:  # Only showcase for one batch per epoch
            break
        
        images = batch['image']
        questions = batch['question_text']
        img_ids = batch['img_id']

        for j in range(min(num_samples, len(images))):
            img_id = img_ids[j]
            question = questions[j]
            image_tensor = images[j].unsqueeze(0).to(device)

            inputs = tokenizer(question, return_tensors="pt", padding='max_length', max_length=512, truncation=True)
            text_input_ids = inputs['input_ids'].to(device)
            text_attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                answer = model.generate_answer(text_input_ids, text_attention_mask, image_tensor)

            print(f"Image ID: {img_id}")
            print(f"Question: {question}")
            print(f"Generated Answer: {answer}")
            print("-" * 40)

# Ensure dataset values are safe during data loading
def safe_transform(image):
    try:
        image = transforms.Resize((224, 224))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        if torch.isnan(image).any() or torch.isinf(image).any():
            print("Found nan or inf in transformed image")
            return None
    except Exception as e:
        print(f"Error in transforming image: {e}")
        return None
    return image

def generate_answer(model, tokenizer, image_path, question, device, max_length=50):
    image = Image.open(image_path).convert('RGB')
    image_tensor = safe_transform(image)
    
    if image_tensor is None:
        return "Invalid Image"

    image_tensor = image_tensor.unsqueeze(0).to(device)

    inputs = tokenizer(question, return_tensors="pt", padding='max_length', max_length=512, truncation=True)
    text_input_ids = inputs['input_ids'].to(device)
    text_attention_mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        answer = model.generate_answer(text_input_ids, text_attention_mask, image_tensor, max_length=max_length)

    return answer if answer else "No Answer Generated"


def evaluate_initial_model(model, eval_dataloader, tokenizer, img_dir, device):
    model.eval()
    print("Initial Evaluation on some QA pairs before training:")

    for i, eval_example in enumerate(eval_dataloader):
        image_path = os.path.join(img_dir, f'COCO_val2014_{eval_example["img_id"][0]:012d}.jpg')
        question = eval_example['question_text'][0]

        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        answer = generate_answer(model, tokenizer, image_path, question, device)

        print(f"Evaluation example {i+1} - Question: {question}")
        print(f"Generated Answer: {answer}")

        if i >= 4:
            break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained BERT and ViT models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    vit_model = prepare_model(chkpt_dir='visual_embed/mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Initialize the multimodal model
    model = MultiModalModel(bert_model, vit_model, tokenizer, vocab_size)
    model.to(device)

    # Freeze the parameters of BERT and ViT models initially
    for param in model.bert_model.parameters():
        param.requires_grad = False
    for param in model.vit_model.parameters():
        param.requires_grad = False

    # Unfreeze BERT and ViT after some epochs
    def unfreeze_model():
        for param in model.bert_model.parameters():
            param.requires_grad = True
        for param in model.vit_model.parameters():
            param.requires_grad = True

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Define the data transformations and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Additional augmentation
        transforms.ColorJitter(),           # Additional augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Training data
    train_img_dir = 'train2014'
    train_annotations_file = 'v2_mscoco_train2014_annotations.json'
    train_questions_file = 'v2_OpenEnded_mscoco_train2014_questions.json'

    train_dataset = CocoVQADataset(img_dir=train_img_dir, annotations_file=train_annotations_file, questions_file=train_questions_file, tokenizer=tokenizer, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Validation data
    val_img_dir = 'val2014'
    val_annotations_file = 'v2_mscoco_val2014_annotations.json'
    val_questions_file = 'v2_OpenEnded_mscoco_val2014_questions.json'
    
    val_dataset = CocoVQADataset(img_dir=val_img_dir, annotations_file=val_annotations_file, questions_file=val_questions_file, tokenizer=tokenizer, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)  # Label smoothing added
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)  # Lowered LR and added weight decay
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = GradScaler()  # Mixed precision scaler

    # Initial evaluation before training
    evaluate_initial_model(model, val_dataloader, tokenizer, val_img_dir, device)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        if epoch == 20:  # Unfreeze the models after 20 epochs
            unfreeze_model()

        avg_loss = train_model(model, train_dataloader, optimizer, criterion, tokenizer, device, scaler)
        scheduler.step()
        print(f"Loss: {avg_loss:.4f}")

        # Showcase predictions after each epoch using val2014 dataset
        showcase_predictions(model, val_dataloader, tokenizer, device, num_samples=5)
        
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
