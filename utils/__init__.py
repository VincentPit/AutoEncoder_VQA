"""
Utility functions for the AutoEncoder VQA project.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
import json
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create attention mask from input IDs."""
    return (input_ids != pad_token_id).float()


def truncate_or_pad_sequence(
    sequence: Union[List, torch.Tensor],
    max_length: int,
    pad_value: Union[int, float] = 0,
    truncate_side: str = "right"
) -> Union[List, torch.Tensor]:
    """Truncate or pad sequence to max_length."""
    if len(sequence) > max_length:
        if truncate_side == "right":
            sequence = sequence[:max_length]
        else:
            sequence = sequence[-max_length:]
    elif len(sequence) < max_length:
        if isinstance(sequence, torch.Tensor):
            pad_tensor = torch.full((max_length - len(sequence),), pad_value, dtype=sequence.dtype)
            sequence = torch.cat([sequence, pad_tensor])
        else:
            sequence = sequence + [pad_value] * (max_length - len(sequence))
    
    return sequence


def calculate_bleu_score(predictions: List[str], targets: List[str]) -> float:
    """Calculate BLEU score for predictions."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothing = SmoothingFunction().method4
        scores = []
        
        for pred, target in zip(predictions, targets):
            pred_tokens = pred.lower().split()
            target_tokens = [target.lower().split()]
            
            score = sentence_bleu(target_tokens, pred_tokens, smoothing_function=smoothing)
            scores.append(score)
        
        return np.mean(scores)
    
    except ImportError:
        logging.warning("NLTK not available for BLEU score calculation")
        return 0.0


def save_predictions(
    predictions: List[str],
    targets: List[str],
    image_ids: List[str],
    filepath: str
) -> None:
    """Save predictions to JSON file."""
    results = [
        {
            'image_id': img_id,
            'prediction': pred,
            'target': target
        }
        for pred, target, img_id in zip(predictions, targets, image_ids)
    ]
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress meter for training."""
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return f'[{fmt}/{fmt.format(num_batches)}]'


def freeze_model_parameters(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze model parameters."""
    for param in model.parameters():
        param.requires_grad = not freeze


def get_parameter_count_by_layer(model: nn.Module) -> Dict[str, int]:
    """Get parameter count for each layer in the model."""
    param_counts = {}
    
    for name, module in model.named_modules():
        if not list(module.children()):  # leaf module
            param_count = sum(p.numel() for p in module.parameters())
            param_counts[name] = param_count
    
    return param_counts


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds %= 60
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.1f}s"