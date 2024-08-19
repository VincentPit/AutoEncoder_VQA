import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from transformers import BertTokenizer
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt

def count_data_entries(csv_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file)
    # Count the number of rows in the DataFrame
    num_entries = len(data)
    return num_entries

class VQADataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Determine the maximum lengths for questions and answers
        self.max_question_length = 0
        self.max_answer_length = 0
        question_lengths = []
        answer_lengths = []
        empty_answers = 0

        for idx in range(len(self.data)):
            question = self.data.iloc[idx, 0]
            answer = self.data.iloc[idx, 1]
            
            if pd.isna(answer) or answer.strip() == "":
                empty_answers += 1

            question_tokenized = tokenizer(question, return_tensors="pt", padding=False, truncation=True)['input_ids'].squeeze()
            answer_tokenized = tokenizer(answer, return_tensors="pt", padding=False, truncation=True)['input_ids'].squeeze()
            
            question_lengths.append(len(question_tokenized))
            answer_lengths.append(len(answer_tokenized))
            
            self.max_question_length = max(self.max_question_length, len(question_tokenized))
            self.max_answer_length = max(self.max_answer_length, len(answer_tokenized))
        
        print(f"Empty answers: {empty_answers}")
        self.plot_length_distribution(question_lengths, 'Question Lengths')
        self.plot_length_distribution(answer_lengths, 'Answer Lengths')

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
        question_tokenized = self.tokenizer(question, return_tensors="pt", padding=False, truncation=True)['input_ids'].squeeze()
        answer_tokenized = self.tokenizer(answer, return_tensors="pt", padding=False, truncation=True)['input_ids'].squeeze()

        # Pad sequences to the maximum lengths determined
        question_padded = self.pad_sequence(question_tokenized, self.max_question_length)
        answer_padded = self.pad_sequence(answer_tokenized, self.max_answer_length)

        sample = {
            'question': question_padded,
            'answer': answer_padded,
            'image': image,
            'question_text': question,
            'answer_text': answer,
            'img_id': img_id
        }
        
        return sample

    def pad_sequence(self, sequence, max_len, padding_value=0):
        if len(sequence) < max_len:
            sequence = torch.cat([sequence, torch.full((max_len - len(sequence),), padding_value)])
        return sequence[:max_len]

    def plot_length_distribution(self, lengths, title):
        counter = Counter(lengths)
        lengths, counts = zip(*counter.items())
        plt.bar(lengths, counts)
        plt.title(title)
        plt.xlabel('Length')
        plt.ylabel('Count')
        plt.show()

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
    print("Max_Question:", dataset.max_question_length)
    print("Max_Answer:", dataset.max_answer_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # Example of iterating through the dataloader and saving samples
    for batch in dataloader:
        save_samples(batch, save_dir)
        break  # Save only the first batch for demonstration
    
    csv_file = 'dataset/data_train.csv'
    num_entries = count_data_entries(csv_file)
    print(f"Number of data entries in the training dataset: {num_entries}")
