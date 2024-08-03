import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from PIL import Image
import json
from cross_model import MultiModalModel
from visual_embed.models import prepare_model
import os

def load_model(model_path, device):
    # Initialize BERT and ViT models
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    vit_model = prepare_model(chkpt_dir='visual_embed/mae_visualize_vit_large.pth', arch='mae_vit_large_patch16', only_encoder=True).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    # Create an instance of the multimodal model
    model = MultiModalModel(bert_model, vit_model, tokenizer, vocab_size)
    model.to(device)

    # Load trained model parameters
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, tokenizer

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
    
    # Debugging statements
    print(f"Device check - image_tensor: {image_tensor.device}")
    print(f"Device check - text_input_ids: {text_input_ids.device}")
    print(f"Device check - text_attention_mask: {text_attention_mask.device}")

    model.eval()
    with torch.no_grad():
        answer = model.generate_answer(text_input_ids, text_attention_mask, image_tensor, max_length=max_length)

    return answer

def save_result(image_path, question, answer, save_dir):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the image
    image = Image.open(image_path)
    image.save(os.path.join(save_dir, 'image.jpg'))
    
    # Save the result as a JSON file
    result = {
        "question": question,
        "answer": answer
    }
    
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'multimodal_model.pth'
    
    print("Model_path:", model_path)
    model, tokenizer = load_model(model_path, device)
    
    # Provide image path and question
    image_path = 'saved_samples/image868.png'
    question = 'what is the colour of the soap where the faucet is facing'
    
    # Generate answer
    answer = generate_answer(model, tokenizer, image_path, question, device)
    
    # Save the result
    save_dir = 'results'
    save_result(image_path, question, answer, save_dir)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Results saved in directory: {save_dir}")
