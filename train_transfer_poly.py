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
import numpy as np

# Import custom models and DataLoader
from transfer_cross_attention_model import MultiModalModel
from coco_dataloader import CocoVQADataset
from visual_embed.models import prepare_model

def train_model(model, dataloader, optimizer, criterion, tokenizer, device, clip_value=1.0):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training Batches"):
        text_input_ids = batch['question'].to(device)
        text_attention_mask = (text_input_ids != tokenizer.pad_token_id).to(device)
        image_tensor = batch['image'].to(device)
        answer = batch['answer'].to(device)

        # Debug: Check for nan or inf in inputs
        if torch.isnan(text_input_ids).any() or torch.isinf(text_input_ids).any():
            print("Found nan or inf in text_input_ids")
        if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
            print("Found nan or inf in image_tensor")
        if torch.isnan(answer).any() or torch.isinf(answer).any():
            print("Found nan or inf in answer")

        optimizer.zero_grad()

        output = model(text_input_ids, text_attention_mask, image_tensor, answer[:, :-1])

        # Shift targets to align with outputs
        target = answer[:, 1:].contiguous()

        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

        # Debug: Check for nan or inf in loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Found nan or inf in loss")

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def generate_answer(model, tokenizer, image_path, question, device, max_length=50):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing image
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    inputs = tokenizer(question, return_tensors="pt", padding='max_length', max_length=512, truncation=True)
    text_input_ids = inputs['input_ids'].to(device)
    text_attention_mask = inputs['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        answer = model.module.generate_answer(text_input_ids, text_attention_mask, image_tensor, max_length=max_length)

    return answer

def evaluate_initial_model(model, eval_dataloader, tokenizer, img_dir, device):
    model.eval()
    print("Initial Evaluation on some QA pairs before training:")

    for i, eval_example in enumerate(eval_dataloader):
        image_path = os.path.join(img_dir, f'COCO_train2014_{eval_example["img_id"][0]:012d}.jpg')
        question = eval_example['question_text'][0]

        # Debug print
        print(f"Checking image path: {image_path}")

        # Check if the file exists
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue  # Skip to the next example

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
    
    print("Number of GPUs available: ", torch.cuda.device_count())
    print("Current GPU: ", torch.cuda.current_device())
    print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load VQA dataset for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing image
    ])

    img_dir = 'train2014'  # Update this path to your images directory
    annotations_file = 'v2_mscoco_train2014_annotations.json'  # Update this path to your annotations file
    questions_file = 'v2_OpenEnded_mscoco_train2014_questions.json'  # Update this path to your questions file
    save_dir = 'saved_samples'

    train_dataset = CocoVQADataset(img_dir=img_dir, annotations_file=annotations_file, questions_file=questions_file, tokenizer=tokenizer, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Load VQA dataset for evaluation
    eval_img_dir = 'train2014'  # Same as training images directory
    eval_annotations_file = 'v2_mscoco_train2014_annotations.json'  # Update this path to your evaluation annotations file
    eval_questions_file = 'v2_OpenEnded_mscoco_train2014_questions.json'  # Update this path to your evaluation questions file
    eval_dataset = CocoVQADataset(img_dir=eval_img_dir, annotations_file=eval_annotations_file, questions_file=eval_questions_file, tokenizer=tokenizer, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Initial evaluation before training
    evaluate_initial_model(model, eval_dataloader, tokenizer, eval_img_dir, device)

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
            image_path = os.path.join(eval_img_dir, f'COCO_train2014_{eval_example["img_id"]:012d}.jpg')
            question = eval_example['question_text']

            answer = generate_answer(model, tokenizer, image_path, question, device)

            print(f"Evaluation example - Question: {question}")
            print(f"Generated Answer: {answer}")

    # Save the trained model
    torch.save(model.module.state_dict(), 'transfer_cross_attention10.pth')
