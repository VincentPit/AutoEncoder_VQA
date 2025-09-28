#!/usr/bin/env python3
"""
Setup script for the AutoEncoder VQA project.

This script sets up the entire project environment, downloads data,
and prepares everything for training and evaluation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True, shell=False):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, check=check, shell=shell, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"Error: Python 3.8+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")


def install_requirements():
    """Install Python requirements."""
    print("Installing Python requirements...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("Warning: requirements.txt not found. Skipping dependency installation.")
        return
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    run_command(cmd)
    print("✓ Requirements installed")


def setup_directories():
    """Create necessary directories."""
    print("Setting up directory structure...")
    
    directories = [
        "data",
        "data/train2014",
        "data/val2014",
        "data/test2014",
        "checkpoints",
        "results",
        "logs",
        "visual_embed",
        "config",
        "scripts",
        "tests",
        "utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")


def download_data(skip_images=False, skip_annotations=False, skip_models=False):
    """Download datasets and models."""
    if Path("scripts/download_data.py").exists():
        print("Downloading data using download script...")
        
        cmd = [sys.executable, "scripts/download_data.py"]
        
        if skip_images:
            cmd.append("--skip_images")
        if skip_annotations:
            cmd.append("--skip_annotations")
        if skip_models:
            cmd.append("--skip_models")
        
        run_command(cmd, check=False)  # Don't exit on error
    else:
        print("Warning: Download script not found. Skipping data download.")


def run_tests():
    """Run the test suite."""
    if Path("tests/test_all.py").exists():
        print("Running tests...")
        
        cmd = [sys.executable, "tests/test_all.py"]
        result = run_command(cmd, check=False)
        
        if result.returncode == 0:
            print("✓ All tests passed")
        else:
            print("⚠ Some tests failed")
    else:
        print("Warning: Test suite not found. Skipping tests.")


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            print(f"  Primary GPU: {gpu_name}")
        else:
            print("⚠ CUDA not available - will use CPU")
    except ImportError:
        print("⚠ PyTorch not installed - cannot check GPU")


def verify_setup():
    """Verify the setup is complete."""
    print("\nVerifying setup...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        "config/config.yaml"
    ]
    
    required_dirs = [
        "models",
        "dataloaders", 
        "visual_embed",
        "question_embed",
        "trainings",
        "scripts",
        "utils"
    ]
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - Missing")
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - Missing")
    
    # Check for data
    data_items = [
        "data/train2014",
        "data/val2014", 
        "visual_embed/mae_visualize_vit_large.pth"
    ]
    
    print("\nData availability:")
    for item in data_items:
        path = Path(item)
        if path.exists():
            if path.is_dir():
                count = len(list(path.iterdir()))
                print(f"✓ {item} ({count} files)")
            else:
                size_mb = path.stat().st_size / (1024**2)
                print(f"✓ {item} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {item} - Not found")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review and update config/config.yaml as needed")
    print("2. If you skipped data download, run:")
    print("   python scripts/download_data.py")
    print("\n3. To train a model:")
    print("   python scripts/train_improved.py --config config/config.yaml")
    print("\n4. To evaluate a trained model:")
    print("   python scripts/evaluate.py --config config/config.yaml --model_path checkpoints/best_model.pth")
    print("\n5. For interactive inference:")
    print("   python scripts/interactive.py --config config/config.yaml --model_path checkpoints/best_model.pth --interactive")
    print("\n6. To run tests:")
    print("   python tests/test_all.py")
    
    print("\nUseful directories:")
    print("- config/: Configuration files")
    print("- data/: Datasets (images, annotations, questions)")
    print("- models/: Model architectures")
    print("- scripts/: Training, evaluation, and utility scripts")
    print("- checkpoints/: Saved model checkpoints")
    print("- results/: Evaluation results and predictions")
    print("- logs/: Training logs")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup AutoEncoder VQA project")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install")
    parser.add_argument("--skip-data", action="store_true", help="Skip data download")
    parser.add_argument("--skip-images", action="store_true", help="Skip image download")
    parser.add_argument("--skip-annotations", action="store_true", help="Skip annotation download") 
    parser.add_argument("--skip-models", action="store_true", help="Skip model download")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--verify-only", action="store_true", help="Only verify setup")
    
    args = parser.parse_args()
    
    print("AutoEncoder VQA Project Setup")
    print("="*40)
    
    if args.verify_only:
        verify_setup()
        return
    
    # Check Python version
    check_python_version()
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    if not args.skip_install:
        install_requirements()
    
    # Check GPU
    check_gpu()
    
    # Download data
    if not args.skip_data:
        download_data(
            skip_images=args.skip_images,
            skip_annotations=args.skip_annotations,
            skip_models=args.skip_models
        )
    
    # Run tests
    if not args.skip_tests:
        run_tests()
    
    # Verify setup
    verify_setup()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()