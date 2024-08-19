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
import numpy as np

# Import custom models and DataLoader
from co_decoder_posi_v4_2 import MultiModalModel
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
            #print("Found nan or inf in model output")
            continue

        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

        # Apply epsilon to avoid NaNs in loss
        loss = loss + epsilon

        # Check for NaN/Inf in the loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            #print("Found nan or inf in loss")
            continue

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Ensure no NaNs in gradients before optimizer step
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    #print(f"Found nan or inf in gradients of {name}, resetting gradient")
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

def beam_search(model, text_input_ids, text_attention_mask, image_tensor, beam_width=5, max_length=50):
    # Start with the initial input
    sequences = [[list(), 1.0]]  # List of sequences, each with a score (log probability)

    for _ in range(max_length):
        all_candidates = list()
        
        for seq, score in sequences:
            input_seq = torch.tensor([seq]).to(text_input_ids.device)  # Convert the sequence to tensor
            input_attention_mask = torch.ones_like(input_seq).to(text_input_ids.device)

            # Prepare model inputs: concatenate the sequence with the input text and image tensor
            input_ids = torch.cat((text_input_ids, input_seq), dim=1)
            attention_mask = torch.cat((text_attention_mask, input_attention_mask), dim=1)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask, image_tensor, input_seq)
                outputs = outputs[:, -1, :]  # Take the last token output

            # Get top-k predictions with their log probabilities
            probs = torch.log_softmax(outputs, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, beam_width, dim=-1)

            # Update the sequences with new candidates
            for i in range(beam_width):
                candidate = [seq + [top_k_indices[0, i].item()], score * top_k_probs[0, i].item()]
                all_candidates.append(candidate)

        # Order all candidates by their score and select the best ones
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

    # Select the sequence with the highest score
    best_seq = sequences[0][0]
    return best_seq


def generate_answer(model, tokenizer, image_path, question, device, beam_width=5, max_length=50):
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
        generated_seq = beam_search(model, text_input_ids, text_attention_mask, image_tensor, beam_width=beam_width, max_length=max_length)

    # Convert generated sequence to tokens
    generated_tokens = tokenizer.decode(generated_seq, skip_special_tokens=True)
    
    return generated_tokens if generated_tokens else "No Answer Generated"

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

        answer = generate_answer(model, tokenizer, image_path, question, device, beam_width=5)

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
    model = MultiModalModel(bert_model, vit_model, tokenizer, vocab_size, dropout_rate=0.3)
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

    val_dataset = CocoVQADataset(img_dir=img_dir, annotations_file=annotations_file, questions_file=questions_file, tokenizer=tokenizer, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Define the loss criterion and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)  # AdamW with weight decay

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Initial model evaluation
    #evaluate_initial_model(model, val_dataloader, tokenizer, img_dir, device)

    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train the model
        avg_loss = train_model(model, train_dataloader, optimizer, criterion, tokenizer, device)
        print(f"Average training loss: {avg_loss:.4f}")

        # Adjust learning rate based on scheduler
        scheduler.step()

        # Showcase some predictions after each epoch
        showcase_predictions(model, val_dataloader, tokenizer, device, num_samples=5, dataset_name="Validation")

    torch.save(model.state_dict(), "checkpoints/v4.2.pth")