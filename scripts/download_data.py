"""
Data download and setup script for VQA training.

This script downloads and prepares the necessary datasets and pre-trained models.
"""

import os
import requests
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import argparse


class DataDownloader:
    """Download and setup VQA datasets and models."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """Initialize downloader."""
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Dataset URLs
        self.dataset_urls = {
            'train_images': 'http://images.cocodataset.org/zips/train2014.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2014.zip',
            'test_images': 'http://images.cocodataset.org/zips/test2014.zip',
            'train_annotations': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
            'val_annotations': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
            'train_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
            'val_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
            'test_questions': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip'
        }
        
        # Model URLs
        self.model_urls = {
            'mae_vit_large': 'https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth'
        }
    
    def download_file(self, url: str, filename: str, extract_to: Optional[str] = None) -> None:
        """Download a file with progress bar."""
        filepath = self.data_dir / filename if extract_to is None else Path(extract_to) / filename
        
        # Check if file already exists
        if filepath.exists():
            print(f"File already exists: {filepath}")
            return
        
        print(f"Downloading {filename}...")
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Downloaded: {filepath}")
        
        # Extract if it's a zip file
        if filename.endswith('.zip') and extract_to:
            self.extract_zip(filepath, Path(extract_to))
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        """Extract zip file."""
        print(f"Extracting {zip_path.name}...")
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to extract
            files = zip_ref.namelist()
            
            with tqdm(desc=f"Extracting {zip_path.name}", total=len(files)) as pbar:
                for file in files:
                    zip_ref.extract(file, extract_to)
                    pbar.update(1)
        
        print(f"Extracted to: {extract_to}")
        
        # Remove zip file to save space
        zip_path.unlink()
        print(f"Removed zip file: {zip_path}")
    
    def download_coco_images(self, splits: list = None) -> None:
        """Download COCO images."""
        if splits is None:
            splits = ['train', 'val']
        
        for split in splits:
            url = self.dataset_urls[f'{split}_images']
            filename = f'{split}2014.zip'
            
            self.download_file(url, filename, extract_to=str(self.data_dir))
    
    def download_vqa_annotations(self, splits: list = None) -> None:
        """Download VQA annotations."""
        if splits is None:
            splits = ['train', 'val']
        
        for split in splits:
            # Download annotations
            ann_url = self.dataset_urls[f'{split}_annotations']
            ann_filename = f'v2_Annotations_{split.title()}_mscoco.zip'
            
            self.download_file(ann_url, ann_filename, extract_to=str(self.data_dir))
            
            # Download questions
            q_url = self.dataset_urls[f'{split}_questions']
            q_filename = f'v2_Questions_{split.title()}_mscoco.zip'
            
            self.download_file(q_url, q_filename, extract_to=str(self.data_dir))
    
    def download_pretrained_models(self) -> None:
        """Download pre-trained models."""
        # Create visual_embed directory
        visual_embed_dir = Path('visual_embed')
        visual_embed_dir.mkdir(exist_ok=True)
        
        # Download MAE ViT Large
        mae_url = self.model_urls['mae_vit_large']
        mae_filename = 'mae_visualize_vit_large.pth'
        
        self.download_file(mae_url, mae_filename, extract_to=str(visual_embed_dir))
    
    def setup_directory_structure(self) -> None:
        """Create the required directory structure."""
        directories = [
            'data/train2014',
            'data/val2014',
            'data/test2014',
            'checkpoints',
            'results',
            'logs',
            'visual_embed'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def create_sample_config(self) -> None:
        """Create a sample configuration file."""
        config = {
            "model": {
                "max_seq_length": 512,
                "dropout_rate": 0.1,
                "num_attention_heads": 8,
                "hidden_size": 768,
                "cross_attention_layers": 4,
                "decoder_layers": 6,
                "vocab_size": 30522,
                "bert_model": "bert-base-uncased",
                "vit_checkpoint": "visual_embed/mae_visualize_vit_large.pth",
                "vit_architecture": "mae_vit_large_patch16"
            },
            "training": {
                "batch_size": 16,
                "num_epochs": 50,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "optimizer": "AdamW",
                "scheduler": "cosine",
                "warmup_steps": 1000,
                "gradient_clip_norm": 1.0,
                "use_amp": True,
                "accumulation_steps": 1,
                "freeze_bert": True,
                "freeze_vit": True,
                "val_interval": 1000,
                "save_interval": 5000,
                "patience": 10,
                "min_delta": 0.001
            },
            "data": {
                "train_images": "data/train2014/",
                "val_images": "data/val2014/",
                "test_images": "data/test2014/",
                "train_annotations": "data/v2_mscoco_train2014_annotations.json",
                "val_annotations": "data/v2_mscoco_val2014_annotations.json",
                "train_questions": "data/v2_OpenEnded_mscoco_train2014_questions.json",
                "val_questions": "data/v2_OpenEnded_mscoco_val2014_questions.json",
                "image_size": 224,
                "num_workers": 4,
                "pin_memory": True,
                "use_augmentation": True,
                "horizontal_flip": True,
                "color_jitter": True
            },
            "inference": {
                "beam_size": 5,
                "max_answer_length": 50,
                "temperature": 1.0,
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": True
            },
            "logging": {
                "use_wandb": False,
                "project_name": "autoencoder_vqa",
                "experiment_name": "default",
                "log_dir": "logs/",
                "log_level": "INFO",
                "metrics": ["loss", "accuracy", "bleu", "cider"]
            },
            "hardware": {
                "device": "auto",
                "use_data_parallel": True,
                "gradient_checkpointing": False
            },
            "paths": {
                "checkpoint_dir": "checkpoints/",
                "best_model_path": "checkpoints/best_model.pth",
                "results_dir": "results/",
                "predictions_file": "results/predictions.json",
                "cache_dir": ".cache/"
            },
            "evaluation": {
                "compute_bleu": True,
                "compute_cider": True,
                "compute_rouge": True,
                "eval_batch_size": 32,
                "save_predictions": True,
                "save_attention_maps": False,
                "num_visualization_samples": 10
            },
            "seed": 42
        }
        
        config_path = Path('config/config.yaml')
        config_path.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Sample configuration created: {config_path}")
    
    def verify_downloads(self) -> Dict[str, bool]:
        """Verify that all required files are downloaded."""
        required_files = [
            'data/train2014',
            'data/val2014',
            'data/v2_mscoco_train2014_annotations.json',
            'data/v2_mscoco_val2014_annotations.json',
            'data/v2_OpenEnded_mscoco_train2014_questions.json',
            'data/v2_OpenEnded_mscoco_val2014_questions.json',
            'visual_embed/mae_visualize_vit_large.pth'
        ]
        
        status = {}
        print("\nVerifying downloads...")
        
        for file_path in required_files:
            path = Path(file_path)
            exists = path.exists()
            status[file_path] = exists
            
            if exists:
                if path.is_dir():
                    num_files = len(list(path.iterdir()))
                    print(f"✓ {file_path} ({num_files} files)")
                else:
                    size = path.stat().st_size / (1024**2)  # MB
                    print(f"✓ {file_path} ({size:.1f} MB)")
            else:
                print(f"✗ {file_path} - Missing")
        
        return status


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download VQA datasets and models")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Models directory")
    parser.add_argument("--skip_images", action="store_true", help="Skip downloading images")
    parser.add_argument("--skip_annotations", action="store_true", help="Skip downloading annotations")
    parser.add_argument("--skip_models", action="store_true", help="Skip downloading models")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], 
                      help="Dataset splits to download")
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DataDownloader(args.data_dir, args.models_dir)
    
    print("Setting up AutoEncoder VQA project...")
    print("="*50)
    
    # Setup directory structure
    downloader.setup_directory_structure()
    
    # Download data
    if not args.skip_images:
        print("\nDownloading COCO images...")
        downloader.download_coco_images(args.splits)
    
    if not args.skip_annotations:
        print("\nDownloading VQA annotations...")
        downloader.download_vqa_annotations(args.splits)
    
    if not args.skip_models:
        print("\nDownloading pre-trained models...")
        downloader.download_pretrained_models()
    
    # Create sample config
    print("\nCreating sample configuration...")
    downloader.create_sample_config()
    
    # Verify downloads
    status = downloader.verify_downloads()
    
    # Summary
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    success_count = sum(status.values())
    total_count = len(status)
    
    print(f"Successfully downloaded: {success_count}/{total_count} items")
    
    if success_count == total_count:
        print("✓ All required files downloaded successfully!")
        print("\nNext steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Update config/config.yaml as needed")
        print("3. Start training: python scripts/train_improved.py --config config/config.yaml")
    else:
        print("✗ Some files are missing. Please check the errors above.")


if __name__ == "__main__":
    main()