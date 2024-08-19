import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import DataLoader

# Import custom models and DataLoader
from co_decoder_posi_v4_2 import MultiModalModel
from coco_dataloader import CocoVQADataset
from visual_embed.models import prepare_model

def get_data_loader(img_dir, annotations_file, questions_file, tokenizer, transform, batch_size=16, num_workers=4):
    dataset = CocoVQADataset(img_dir=img_dir, annotations_file=annotations_file, questions_file=questions_file, tokenizer=tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def test_model(model, dataloader, tokenizer, device):
    model.eval()
    
    for batch in dataloader:
        text_input_ids = batch['question'].to(device)
        text_attention_mask = (text_input_ids != tokenizer.pad_token_id).to(device)
        
        image_tensor = batch['image'].to(device)
        answer = batch['answer'].to(device)
        
        with torch.no_grad():
            output = model(text_input_ids, text_attention_mask, image_tensor, answer[:, :-1])
            print(f"Model output shape: {output.shape}")
            break  # Test only the first batch for simplicity

def showcase_predictions(model, dataloader, tokenizer, device, num_samples=5):
    model.eval()
    print("\nShowcasing Model Predictions:")
    
    for batch in dataloader:
        images = batch['image']
        questions = batch['question_text']
        img_ids = batch['img_id']
        
        for i in range(min(num_samples, len(images))):
            img_id = img_ids[i]
            question = questions[i]
            image_tensor = images[i].unsqueeze(0).to(device)
            
            inputs = tokenizer(question, return_tensors="pt", padding='max_length', max_length=512, truncation=True)
            text_input_ids = inputs['input_ids'].to(device)
            text_attention_mask = inputs['attention_mask'].to(device)
            
            with torch.no_grad():
                answer = model.generate_answer(text_input_ids, text_attention_mask, image_tensor)
                
            print(f"Image ID: {img_id}")
            print(f"Question: {question}")
            print(f"Generated Answer: {answer}")
            print("-" * 40)
            
        break  # Showcase only one batch for simplicity

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

    # Define the data transformations and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing image
    ])

    img_dir = 'train2014'
    annotations_file = 'v2_mscoco_train2014_annotations.json'
    questions_file = 'v2_OpenEnded_mscoco_train2014_questions.json'

    dataloader = get_data_loader(img_dir, annotations_file, questions_file, tokenizer, transform)

    # Test model forward pass
    test_model(model, dataloader, tokenizer, device)
    
    # Showcase predictions
    showcase_predictions(model, dataloader, tokenizer, device, num_samples=5)
    
    print("Testing complete. No training performed.")
