import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt

def count_data_entries(csv_file):
    data = pd.read_csv(csv_file)
    return len(data)

def count_questions_entries(questions_file):
    with open(questions_file, 'r') as f:
        data = json.load(f)
    return len(data['questions'])

class CocoVQADataset(Dataset):
    def __init__(self, img_dir, annotations_file, questions_file, tokenizer, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer

        # Load annotations and questions
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        with open(questions_file, 'r') as f:
            self.questions = json.load(f)['questions']
        
        # Create a dictionary to map question_id to its corresponding question
        self.question_dict = {q['question_id']: q for q in self.questions}
        
        # Determine the maximum lengths for questions and answers
        self.max_question_length = 0
        self.max_answer_length = 0
        question_lengths = []
        answer_lengths = []

        for annotation in self.annotations:
            question_id = annotation['question_id']
            question = self.question_dict[question_id]['question']
            answer = annotation['multiple_choice_answer']

            question_tokenized = tokenizer(question, return_tensors="pt", padding=False, truncation=True)['input_ids'].squeeze()
            answer_tokenized = tokenizer(answer, return_tensors="pt", padding=False, truncation=True)['input_ids'].squeeze()

            question_lengths.append(len(question_tokenized))
            answer_lengths.append(len(answer_tokenized))
            
            self.max_question_length = max(self.max_question_length, len(question_tokenized))
            self.max_answer_length = max(self.max_answer_length, len(answer_tokenized))

        print(f"Max Question Length: {self.max_question_length}")
        print(f"Max Answer Length: {self.max_answer_length}")

        self.plot_length_distribution(question_lengths, 'Question Lengths')
        self.plot_length_distribution(answer_lengths, 'Answer Lengths')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_id = annotation['image_id']
        question_id = annotation['question_id']

        # Get the corresponding question and answer
        question = self.question_dict[question_id]['question']
        answer = annotation['multiple_choice_answer']

        # Load and process the image
        img_name = os.path.join(self.img_dir, f'COCO_train2014_{img_id:012d}.jpg')
        image = Image.open(img_name).convert('RGB')

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
    img_dir = 'train2014'  # Update this path to your images directory
    annotations_file = 'v2_mscoco_train2014_annotations.json'  # Update this path to your annotations file
    questions_file = 'v2_OpenEnded_mscoco_train2014_questions.json'  # Update this path to your questions file
    save_dir = 'saved_samples'

    dataset = CocoVQADataset(img_dir=img_dir, annotations_file=annotations_file, questions_file=questions_file, tokenizer=tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # Example of iterating through the dataloader and saving samples
    for batch in dataloader:
        # Check if CLS and SEP tokens are present
        for i in range(len(batch['question'])):
            question_ids = batch['question'][i]
            answer_ids = batch['answer'][i]
            
            # Decode token IDs back to text
            question_text = tokenizer.decode(question_ids, skip_special_tokens=False)
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
            
            # Check for special tokens
            cls_token = tokenizer.cls_token
            sep_token = tokenizer.sep_token
            
            cls_in_question = cls_token in question_text
            sep_in_question = sep_token in question_text
            cls_in_answer = cls_token in answer_text
            sep_in_answer = sep_token in answer_text
            
            print(f"Question Text: {question_text}")
            print(f"Answer Text: {answer_text}")
            print(f"CLS in Question: {cls_in_question}, SEP in Question: {sep_in_question}")
            print(f"CLS in Answer: {cls_in_answer}, SEP in Answer: {sep_in_answer}")

        save_samples(batch, save_dir)
        break  # Save only the first batch for demonstration
    
    num_entries = count_questions_entries(questions_file)
    print(f"Number of question entries in the questions file: {num_entries}")
