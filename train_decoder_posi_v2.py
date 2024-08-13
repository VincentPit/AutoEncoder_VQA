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
from co_decoder_posi_v2 import MultiModalModel
from coco_dataloader import CocoVQADataset
from visual_embed.models import prepare_model

def train_model(model, dataloader, optimizer, criterion, tokenizer, device, clip_value=1.0, epsilon=1e-8):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training Batches"):
        if not validate_batch(batch):
            continue

        text_input_ids = batch['question'].to(device)
        text_attention_mask = (text_input_ids != tokenizer.pad_token_id).to(device)
        image_tensor = batch['image'].to(device)
        answer = batch['answer'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(text_input_ids, text_attention_mask, image_tensor, answer[:, :-1])

        target = answer[:, 1:].contiguous()

        # Check for NaN/Inf in the model output
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Found nan or inf in model output")
            continue

        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

        # Apply epsilon to avoid NaNs in loss
        loss = loss + epsilon

        # Check for NaN/Inf in the loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Found nan or inf in loss")
            continue

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Ensure no NaNs in gradients before optimizer step
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Found nan or inf in gradients of {name}, resetting gradient")
                    param.grad.data.zero_()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_batch(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):  # Ensure the value is a tensor before checking
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"Warning: {key} contains nan or inf values.")
                return False
    return True

def showcase_predictions(model, dataloader, tokenizer, device, num_samples=5, dataset_name="Evaluation"):
    model.eval()
    print(f"\nShowcasing Model Predictions on {dataset_name} set:")
    
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
        image_path = os.path.join(img_dir, f'COCO_train2014_{eval_example["img_id"][0]:012d}.jpg')
        question = eval_example['question_text'][0]

        # Debug print
        print(f"Checking image path: {image_path}")

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

    # Freeze the parameters of BERT and ViT models
    for param in model.bert_model.parameters():
        param.requires_grad = False
    for param in model.vit_model.parameters():
        param.requires_grad = False

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Define the data transformations and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing image
    ])

    img_dir = 'train2014'
    annotations_file = 'v2_mscoco_train2014_annotations.json'
    questions_file = 'v2_OpenEnded_mscoco_train2014_questions.json'

    train_dataset = CocoVQADataset(img_dir=img_dir, annotations_file=annotations_file, questions_file=questions_file, tokenizer=tokenizer, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    eval_dataset = CocoVQADataset(img_dir=img_dir, annotations_file=annotations_file, questions_file=questions_file, tokenizer=tokenizer, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Initial evaluation before training
    evaluate_initial_model(model, eval_dataloader, tokenizer, img_dir, device)

    # Showcase predictions before training begins
    showcase_predictions(model, train_dataloader, tokenizer, device, num_samples=5, dataset_name="Train2014")
    showcase_predictions(model, eval_dataloader, tokenizer, device, num_samples=5, dataset_name="Eval")

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_model(model, train_dataloader, optimizer, criterion, tokenizer, device)
        scheduler.step()
        print(f"Loss: {avg_loss:.4f}")

        # Showcase predictions from the train2014 dataset after each epoch
        showcase_predictions(model, train_dataloader, tokenizer, device, num_samples=5, dataset_name="Train2014")

        # Showcase predictions from the eval dataset after each epoch
        showcase_predictions(model, eval_dataloader, tokenizer, device, num_samples=5, dataset_name="Eval")

    # Save the model after training
    torch.save(model.state_dict(), "multimodal_model.pth")
