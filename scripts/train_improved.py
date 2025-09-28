"""
Improved training script for the VQA model.

This script provides a comprehensive training pipeline with:
- Configuration management
- Model checkpointing
- Validation and evaluation
- Logging and monitoring
- Mixed precision training
"""

import os
import yaml
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import BertModel, BertTokenizer
from torchvision import transforms
from tqdm import tqdm
import wandb
from pathlib import Path

# Local imports
from models.improved_multimodal_model import ImprovedMultiModalModel
from dataloaders.coco_dataloader import CocoVQADataset
from visual_embed.models import prepare_model
from utils import (
    setup_logging, count_parameters, save_checkpoint, load_checkpoint,
    AverageMeter, ProgressMeter, set_random_seed, get_device, format_time
)


class VQATrainer:
    """Comprehensive trainer for VQA models."""
    
    def __init__(self, config: dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = get_device()
        self.logger = setup_logging(
            config['logging']['log_level'],
            config['logging']['log_dir']
        )
        
        # Set random seed for reproducibility
        set_random_seed(config.get('seed', 42))
        
        # Initialize model, optimizer, and other components
        self._init_model()
        self._init_optimizer()
        self._init_data_loaders()
        self._init_training_components()
        
        # Initialize wandb if enabled
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project_name'],
                name=config['logging']['experiment_name'],
                config=config
            )
    
    def _init_model(self) -> None:
        """Initialize the VQA model."""
        self.logger.info("Initializing model...")
        
        # Load pre-trained models
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
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Freeze components if specified
        if self.config['training']['freeze_bert']:
            self.model.freeze_component('bert_model')
            self.logger.info("BERT parameters frozen")
            
        if self.config['training']['freeze_vit']:
            self.model.freeze_component('vit_model')
            self.logger.info("ViT parameters frozen")
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Log model info
        model_size = self.model.get_model_size() if not isinstance(self.model, nn.DataParallel) else self.model.module.get_model_size()
        self.logger.info(f"Model parameters: {model_size}")
    
    def _init_optimizer(self) -> None:
        """Initialize optimizer and scheduler."""
        optimizer_name = self.config['training']['optimizer'].lower()
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Initialize scheduler
        scheduler_name = self.config['training']['scheduler'].lower()
        
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['step_size'],
                gamma=self.config['training']['gamma']
            )
        else:
            self.scheduler = None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config['training']['use_amp'] else None
    
    def _init_data_loaders(self) -> None:
        """Initialize data loaders."""
        self.logger.info("Initializing data loaders...")
        
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get tokenizer from model
        tokenizer = self.model.tokenizer if not isinstance(self.model, nn.DataParallel) else self.model.module.tokenizer
        
        # Training dataset
        train_dataset = CocoVQADataset(
            img_dir=self.config['data']['train_images'],
            annotations_file=self.config['data']['train_annotations'],
            questions_file=self.config['data']['train_questions'],
            tokenizer=tokenizer,
            transform=transform
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        # Validation dataset
        val_dataset = CocoVQADataset(
            img_dir=self.config['data']['val_images'],
            annotations_file=self.config['data']['val_annotations'],
            questions_file=self.config['data']['val_questions'],
            tokenizer=tokenizer,
            transform=transform
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['evaluation']['eval_batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
    
    def _init_training_components(self) -> None:
        """Initialize training components."""
        # Loss criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is pad token
        
        # Metrics
        self.train_loss_meter = AverageMeter('Train Loss', ':.4f')
        self.val_loss_meter = AverageMeter('Val Loss', ':.4f')
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(self.config['paths']['checkpoint_dir'], exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        self.train_loss_meter.reset()
        
        progress = ProgressMeter(
            len(self.train_loader),
            [self.train_loss_meter],
            prefix=f"Epoch [{self.current_epoch}/{self.config['training']['num_epochs']}]"
        )
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move batch to device
            text_input_ids = batch['question'].to(self.device)
            text_attention_mask = (text_input_ids != 0).float().to(self.device)
            image_tensor = batch['image'].to(self.device)
            target_ids = batch['answer'].to(self.device)
            
            # Create decoder input (shift target by one position)
            decoder_input_ids = target_ids[:, :-1]
            target_output_ids = target_ids[:, 1:]
            
            # Forward pass with mixed precision if enabled
            if self.scaler is not None:
                with autocast():
                    logits = self.model(
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        image_tensor=image_tensor,
                        decoder_input_ids=decoder_input_ids
                    )
                    
                    loss = self.model.compute_loss(logits, target_output_ids) if not isinstance(self.model, nn.DataParallel) else self.model.module.compute_loss(logits, target_output_ids)
            else:
                logits = self.model(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    image_tensor=image_tensor,
                    decoder_input_ids=decoder_input_ids
                )
                
                loss = self.model.compute_loss(logits, target_output_ids) if not isinstance(self.model, nn.DataParallel) else self.model.module.compute_loss(logits, target_output_ids)
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            self.train_loss_meter.update(loss.item(), text_input_ids.size(0))
            
            # Log progress
            if batch_idx % 100 == 0:
                progress.display(batch_idx)
        
        return self.train_loss_meter.avg
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        self.val_loss_meter.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                text_input_ids = batch['question'].to(self.device)
                text_attention_mask = (text_input_ids != 0).float().to(self.device)
                image_tensor = batch['image'].to(self.device)
                target_ids = batch['answer'].to(self.device)
                
                # Create decoder input
                decoder_input_ids = target_ids[:, :-1]
                target_output_ids = target_ids[:, 1:]
                
                # Forward pass
                logits = self.model(
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    image_tensor=image_tensor,
                    decoder_input_ids=decoder_input_ids
                )
                
                loss = self.model.compute_loss(logits, target_output_ids) if not isinstance(self.model, nn.DataParallel) else self.model.module.compute_loss(logits, target_output_ids)
                
                # Update metrics
                self.val_loss_meter.update(loss.item(), text_input_ids.size(0))
        
        return self.val_loss_meter.avg
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch + 1
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            self.logger.info(
                f"Epoch {self.current_epoch}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Log to wandb
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': self.current_epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=self.current_epoch,
                    loss=val_loss,
                    filepath=self.config['paths']['best_model_path'],
                    config=self.config
                )
            else:
                self.patience_counter += 1
            
            # Regular checkpoint saving
            if self.current_epoch % self.config['training']['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['paths']['checkpoint_dir'],
                    f'checkpoint_epoch_{self.current_epoch}.pth'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=self.current_epoch,
                    loss=val_loss,
                    filepath=checkpoint_path,
                    config=self.config
                )
            
            # Early stopping
            if self.patience_counter >= self.config['training']['patience']:
                self.logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
                break
        
        self.logger.info("Training completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train VQA Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = VQATrainer(config)
    
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            trainer.model,
            trainer.optimizer,
            trainer.device
        )
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from epoch {trainer.current_epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()