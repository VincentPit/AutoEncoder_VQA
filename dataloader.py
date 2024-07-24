import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os

class VQADataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        question = self.data.iloc[idx, 0]
        answer = self.data.iloc[idx, 1]
        img_id = self.data.iloc[idx, 2]

        img_path = os.path.join(self.img_dir, f'{img_id}.png')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Tokenize the question and answer
        question_tokenized = self.tokenizer(question, return_tensors="pt", padding='max_length', max_length=512, truncation=True)
        answer_tokenized = self.tokenizer(answer, return_tensors="pt", padding='max_length', max_length=512, truncation=True)

        sample = {
            'question': question_tokenized['input_ids'].squeeze(), 
            'answer': answer_tokenized['input_ids'].squeeze(), 
            'image': image
        }
        return sample
    
    
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    csv_file = 'dataset/data_train.csv'
    img_dir = 'dataset/images'

    dataset = VQADataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Example of iterating through the dataloader
    for batch in dataloader:
        questions = batch['question']
        answers = batch['answer']
        images = batch['image']
        # Training logic here
