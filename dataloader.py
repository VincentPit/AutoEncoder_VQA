import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from transformers import BertTokenizer
from torchvision import transforms

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

        # Define max length for padding
        max_length = 32  # Adjust as needed for your data

        # Tokenize the question and answer
        question_tokenized = self.tokenizer(question, return_tensors="pt", padding=False, max_length=max_length, truncation=True)
        answer_tokenized = self.tokenizer(answer, return_tensors="pt", padding=False, max_length=max_length, truncation=True)

        # Pad sequences to the fixed max length
        question_padded = pad_sequence(question_tokenized['input_ids'].squeeze(), max_length)
        answer_padded = pad_sequence(answer_tokenized['input_ids'].squeeze(), max_length)

        sample = {
            'question': question_padded,
            'answer': answer_padded,
            'image': image,
            'question_text': question,
            'answer_text': answer,
            'img_id': img_id
        }
        return sample
    

def pad_sequence(sequence, max_length, padding_value=0):
    padded_sequence = torch.full((max_length,), padding_value)
    length = min(len(sequence), max_length)
    padded_sequence[:length] = sequence[:length]
    return padded_sequence


def save_samples(batch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(batch['image'])):
        image = transforms.ToPILImage()(batch['image'][i])
        question = batch['question_text'][i]
        answer = batch['answer_text'][i]
        img_id = batch['img_id'][i]

        image.save(os.path.join(save_dir, f'{img_id}.png'))

        with open(os.path.join(save_dir, f'{img_id}.txt'), 'w') as f:
            f.write(f'Question: {question}\nAnswer: {answer}\n')

if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    csv_file = 'dataset/data_train.csv'
    img_dir = 'dataset/images'
    save_dir = 'saved_samples'

    dataset = VQADataset(csv_file=csv_file, img_dir=img_dir, tokenizer=tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Example of iterating through the dataloader and saving samples
    for batch in dataloader:
        save_samples(batch, save_dir)
        break  # Save only the first batch for demonstration
