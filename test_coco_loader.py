import os
import torch
from torchvision import transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from coco_dataloader import CocoVQADataset

# Reuse CocoVQADataset class and save_samples function from your existing code

def check_and_save_pairs(dataloader, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    
    # Iterate through the dataloader
    for batch in dataloader:
        images = batch['image']
        questions = batch['question_text']
        answers = batch['answer_text']
        img_ids = batch['img_id']
        
        # Process a subset of samples
        for i in range(min(num_samples, len(images))):
            img_id = img_ids[i]
            image = images[i]
            question = questions[i]
            answer = answers[i]
            print(question, answer)
            
            # Save the image
            img_path = os.path.join(save_dir, f'{img_id}.png')
            image = transforms.ToPILImage()(image)
            image.save(img_path)
            
            # Save the question and answer
            qa_path = os.path.join(save_dir, f'{img_id}.txt')
            with open(qa_path, 'w') as f:
                f.write(f'Question: {question}\nAnswer: {answer}\n')

        # Exit after saving the desired number of samples
        break

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
    
    # Check and save a subset of image-question-answer pairs
    check_and_save_pairs(dataloader, save_dir, num_samples=5)
