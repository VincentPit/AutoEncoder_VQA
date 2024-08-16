import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
from transformers import BertTokenizer

class COCOMatchDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, transform=None, max_length=512):
        """
        Args:
            image_dir (string): Directory with all the images.
            caption_file (string): Path to the COCO captions file.
            tokenizer (BertTokenizer): Tokenizer for processing text captions.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_length (int): Maximum sequence length for text inputs.
        """
        self.image_dir = image_dir
        self.coco = COCO(caption_file)
        self.ids = list(self.coco.imgs.keys())
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        caption_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = self.coco.loadAnns(caption_ids)
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get the first caption for simplicity (you can extend this to use multiple captions)
        caption = captions[0]['caption']
        
        # Tokenize the caption
        tokens = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        input_ids = tokens['input_ids'].squeeze()  # Shape: [max_length]
        attention_mask = tokens['attention_mask'].squeeze()  # Shape: [max_length]

        return image, input_ids, attention_mask

# Example usage
if __name__ == "__main__":
    # Paths to the dataset directories and annotation files
    image_dir = 'coco_dataset/train2017'
    caption_file = 'coco_dataset/annotations/captions_train2017.json'

    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset and DataLoader
    coco_match_dataset = COCOMatchDataset(image_dir=image_dir, caption_file=caption_file, tokenizer=tokenizer, transform=transform)
    coco_match_dataloader = DataLoader(coco_match_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Iterate over the DataLoader
    for images, input_ids, attention_masks in coco_match_dataloader:
        print("Images batch shape:", images.shape)  # Batch of images
        print("Input IDs batch shape:", input_ids.shape)  # Tokenized captions
        print("Attention masks batch shape:", attention_masks.shape)  # Attention masks
        # You can now pass these to your model
