"""
Interactive inference script for VQA models.

This script allows interactive testing of trained VQA models with custom images and questions.
"""

import os
import argparse
import yaml
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertModel, BertTokenizer

# Local imports
from models.improved_multimodal_model import ImprovedMultiModalModel
from visual_embed.models import prepare_model
from utils import load_checkpoint, get_device


class VQAInference:
    """Interactive inference for VQA models."""
    
    def __init__(self, config: dict, model_path: str):
        """Initialize inference engine."""
        self.config = config
        self.device = get_device()
        
        # Load model
        self._load_model(model_path)
        
        # Initialize data transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Inference engine initialized!")
        print(f"Device: {self.device}")
    
    def _load_model(self, model_path: str) -> None:
        """Load trained model."""
        print("Loading model...")
        
        # Load pre-trained components
        bert_model = BertModel.from_pretrained(self.config['model']['bert_model'])
        vit_model = prepare_model(
            chkpt_dir=self.config['model']['vit_checkpoint'],
            arch=self.config['model']['vit_architecture'],
            only_encoder=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.config['model']['bert_model'])
        
        # Create model
        self.model = ImprovedMultiModalModel(
            bert_model=bert_model,
            vit_model=vit_model,
            tokenizer=self.tokenizer,
            config=self.config['model']
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(model_path, self.model, device=self.device)
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def preprocess_question(self, question: str) -> tuple:
        """Preprocess question for inference."""
        # Tokenize question
        encoded = self.tokenizer(
            question,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['model']['max_seq_length']
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        return input_ids, attention_mask
    
    def generate_answer(
        self,
        image_path: str,
        question: str,
        beam_size: int = None,
        max_length: int = None,
        temperature: float = None
    ) -> str:
        """Generate answer for given image and question."""
        # Use default values if not provided
        beam_size = beam_size or self.config['inference']['beam_size']
        max_length = max_length or self.config['inference']['max_answer_length']
        temperature = temperature or self.config['inference']['temperature']
        
        # Preprocess inputs
        image_tensor = self.preprocess_image(image_path)
        input_ids, attention_mask = self.preprocess_question(question)
        
        # Generate answer
        with torch.no_grad():
            answer = self.model.generate_answer(
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                image_tensor=image_tensor,
                max_length=max_length,
                beam_size=beam_size,
                temperature=temperature
            )
        
        return answer.strip()
    
    def interactive_mode(self) -> None:
        """Run interactive inference mode."""
        print("\n" + "="*50)
        print("INTERACTIVE VQA INFERENCE")
        print("="*50)
        print("Enter 'quit' to exit")
        print("Commands:")
        print("  /help - Show this help message")
        print("  /image <path> - Set image path")
        print("  /beam <size> - Set beam size")
        print("  /temp <value> - Set temperature")
        print("  /maxlen <length> - Set max answer length")
        print("-"*50)
        
        # Default settings
        current_image = None
        beam_size = self.config['inference']['beam_size']
        temperature = self.config['inference']['temperature']
        max_length = self.config['inference']['max_answer_length']
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if user_input.startswith('/'):
                    # Handle commands
                    parts = user_input[1:].split(maxsplit=1)
                    command = parts[0].lower()
                    
                    if command == 'help':
                        print("Commands:")
                        print("  /help - Show this help message")
                        print("  /image <path> - Set image path")
                        print("  /beam <size> - Set beam size")
                        print("  /temp <value> - Set temperature")
                        print("  /maxlen <length> - Set max answer length")
                        print(f"Current settings:")
                        print(f"  Image: {current_image or 'Not set'}")
                        print(f"  Beam size: {beam_size}")
                        print(f"  Temperature: {temperature}")
                        print(f"  Max length: {max_length}")
                    
                    elif command == 'image':
                        if len(parts) > 1:
                            image_path = parts[1]
                            if os.path.exists(image_path):
                                current_image = image_path
                                print(f"Image set to: {image_path}")
                            else:
                                print(f"Error: Image not found - {image_path}")
                        else:
                            print("Usage: /image <path>")
                    
                    elif command == 'beam':
                        if len(parts) > 1:
                            try:
                                beam_size = int(parts[1])
                                print(f"Beam size set to: {beam_size}")
                            except ValueError:
                                print("Error: Beam size must be an integer")
                        else:
                            print("Usage: /beam <size>")
                    
                    elif command == 'temp':
                        if len(parts) > 1:
                            try:
                                temperature = float(parts[1])
                                print(f"Temperature set to: {temperature}")
                            except ValueError:
                                print("Error: Temperature must be a number")
                        else:
                            print("Usage: /temp <value>")
                    
                    elif command == 'maxlen':
                        if len(parts) > 1:
                            try:
                                max_length = int(parts[1])
                                print(f"Max length set to: {max_length}")
                            except ValueError:
                                print("Error: Max length must be an integer")
                        else:
                            print("Usage: /maxlen <length>")
                    
                    else:
                        print(f"Unknown command: {command}")
                
                else:
                    # Treat as question
                    if current_image is None:
                        print("Error: No image set. Use /image <path> to set an image first.")
                        continue
                    
                    question = user_input
                    
                    print(f"Image: {current_image}")
                    print(f"Question: {question}")
                    print("Generating answer...")
                    
                    try:
                        answer = self.generate_answer(
                            current_image,
                            question,
                            beam_size=beam_size,
                            max_length=max_length,
                            temperature=temperature
                        )
                        print(f"Answer: {answer}")
                    
                    except Exception as e:
                        print(f"Error generating answer: {str(e)}")
            
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {str(e)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Interactive VQA Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize inference engine
    inference_engine = VQAInference(config, args.model_path)
    
    if args.interactive:
        # Run interactive mode
        inference_engine.interactive_mode()
    
    elif args.image and args.question:
        # Single inference
        print(f"Image: {args.image}")
        print(f"Question: {args.question}")
        print("Generating answer...")
        
        answer = inference_engine.generate_answer(args.image, args.question)
        print(f"Answer: {answer}")
    
    else:
        print("Error: Either use --interactive or provide both --image and --question")
        print("Use --help for more information")


if __name__ == "__main__":
    main()