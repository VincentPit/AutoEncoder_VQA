"""
Evaluation script for VQA models.

This script evaluates trained models on test datasets and computes various metrics.
"""

import os
import json
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# Local imports
from models.improved_multimodal_model import ImprovedMultiModalModel
from dataloaders.coco_dataloader import CocoVQADataset
from visual_embed.models import prepare_model
from utils import load_checkpoint, get_device, calculate_bleu_score, save_predictions


class VQAEvaluator:
    """Evaluator for VQA models."""
    
    def __init__(self, config: dict, model_path: str):
        """Initialize evaluator."""
        self.config = config
        self.device = get_device()
        
        # Load model
        self._load_model(model_path)
        
        # Initialize data loader
        self._init_data_loader()
        
        # Metrics storage
        self.predictions = []
        self.targets = []
        self.image_ids = []
    
    def _load_model(self, model_path: str) -> None:
        """Load trained model from checkpoint."""
        print("Loading model...")
        
        # Load pre-trained components
        bert_model = BertModel.from_pretrained(self.config['model']['bert_model'])
        vit_model = prepare_model(
            chkpt_dir=self.config['model']['vit_checkpoint'],
            arch=self.config['model']['vit_architecture'],
            only_encoder=True
        )
        tokenizer = BertTokenizer.from_pretrained(self.config['model']['bert_model'])
        
        # Create model
        self.model = ImprovedMultiModalModel(
            bert_model=bert_model,
            vit_model=vit_model,
            tokenizer=tokenizer,
            config=self.config['model']
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(model_path, self.model, device=self.device)
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
    
    def _init_data_loader(self) -> None:
        """Initialize test data loader."""
        print("Initializing data loader...")
        
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get tokenizer
        tokenizer = self.model.tokenizer if not hasattr(self.model, 'module') else self.model.module.tokenizer
        
        # Test dataset
        test_dataset = CocoVQADataset(
            img_dir=self.config['data'].get('test_images', self.config['data']['val_images']),
            annotations_file=self.config['data'].get('test_annotations', self.config['data']['val_annotations']),
            questions_file=self.config['data'].get('test_questions', self.config['data']['val_questions']),
            tokenizer=tokenizer,
            transform=transform
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['evaluation']['eval_batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=False
        )
        
        print(f"Test samples: {len(test_dataset)}")
    
    def evaluate(self) -> dict:
        """Evaluate model and compute metrics."""
        print("Starting evaluation...")
        
        self.predictions = []
        self.targets = []
        self.image_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move batch to device
                text_input_ids = batch['question'].to(self.device)
                text_attention_mask = (text_input_ids != 0).float().to(self.device)
                image_tensor = batch['image'].to(self.device)
                
                # Generate answers
                batch_size = text_input_ids.size(0)
                
                for i in range(batch_size):
                    # Generate answer for each sample in batch
                    answer = self.model.generate_answer(
                        text_input_ids[i:i+1],
                        text_attention_mask[i:i+1],
                        image_tensor[i:i+1],
                        max_length=self.config['inference']['max_answer_length'],
                        beam_size=self.config['inference']['beam_size'],
                        temperature=self.config['inference']['temperature']
                    ) if not hasattr(self.model, 'module') else self.model.module.generate_answer(
                        text_input_ids[i:i+1],
                        text_attention_mask[i:i+1],
                        image_tensor[i:i+1],
                        max_length=self.config['inference']['max_answer_length'],
                        beam_size=self.config['inference']['beam_size'],
                        temperature=self.config['inference']['temperature']
                    )
                    
                    self.predictions.append(answer)
                    self.targets.append(batch['answer_text'][i])
                    self.image_ids.append(batch['img_id'][i])
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Save predictions if requested
        if self.config['evaluation']['save_predictions']:
            save_predictions(
                self.predictions,
                self.targets,
                self.image_ids,
                self.config['paths']['predictions_file']
            )
        
        return metrics
    
    def _compute_metrics(self) -> dict:
        """Compute evaluation metrics."""
        print("Computing metrics...")
        
        metrics = {}
        
        # Exact match accuracy
        exact_matches = sum(1 for pred, target in zip(self.predictions, self.targets) 
                          if pred.lower().strip() == target.lower().strip())
        metrics['exact_match_accuracy'] = exact_matches / len(self.predictions)
        
        # BLEU score
        if self.config['evaluation']['compute_bleu']:
            metrics['bleu_score'] = calculate_bleu_score(self.predictions, self.targets)
        
        # Answer length statistics
        pred_lengths = [len(pred.split()) for pred in self.predictions]
        target_lengths = [len(target.split()) for target in self.targets]
        
        metrics['avg_pred_length'] = np.mean(pred_lengths)
        metrics['avg_target_length'] = np.mean(target_lengths)
        
        # Answer type analysis (for VQA datasets)
        answer_types = defaultdict(list)
        for pred, target in zip(self.predictions, self.targets):
            if target.lower() in ['yes', 'no']:
                answer_types['yes/no'].append(pred.lower() == target.lower())
            elif target.isdigit():
                answer_types['number'].append(pred.strip() == target)
            else:
                answer_types['other'].append(pred.lower().strip() == target.lower().strip())
        
        for answer_type, correct_list in answer_types.items():
            if correct_list:
                metrics[f'accuracy_{answer_type}'] = np.mean(correct_list)
                metrics[f'count_{answer_type}'] = len(correct_list)
        
        return metrics
    
    def print_examples(self, num_examples: int = 10) -> None:
        """Print some example predictions."""
        print(f"\n--- Example Predictions (first {num_examples}) ---")
        
        for i in range(min(num_examples, len(self.predictions))):
            print(f"\nExample {i+1}:")
            print(f"Image ID: {self.image_ids[i]}")
            print(f"Prediction: {self.predictions[i]}")
            print(f"Target: {self.targets[i]}")
            print(f"Match: {self.predictions[i].lower().strip() == self.targets[i].lower().strip()}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate VQA Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update output paths
    config['paths']['predictions_file'] = os.path.join(args.output_dir, 'predictions.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = VQAEvaluator(config, args.model_path)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Show examples
    evaluator.print_examples()


if __name__ == "__main__":
    main()