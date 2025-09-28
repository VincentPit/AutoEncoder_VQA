"""
Comprehensive test suite for the AutoEncoder VQA project.
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path
import json

# Import modules to test
from models.positional_embedding import PositionalEmbedding, create_causal_mask, create_padding_mask
from utils import (
    count_parameters, save_checkpoint, load_checkpoint, 
    AverageMeter, truncate_or_pad_sequence, set_random_seed
)


class TestPositionalEmbedding(unittest.TestCase):
    """Test cases for positional embedding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.max_seq_length = 128
        self.d_model = 64
        self.pos_emb = PositionalEmbedding(self.max_seq_length, self.d_model)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.pos_emb.max_seq_length, self.max_seq_length)
        self.assertEqual(self.pos_emb.d_model, self.d_model)
        self.assertEqual(self.pos_emb.pe.shape, (1, self.max_seq_length, self.d_model))
    
    def test_odd_d_model_error(self):
        """Test error for odd d_model."""
        with self.assertRaises(ValueError):
            PositionalEmbedding(100, 63)  # Odd d_model should raise error
    
    def test_forward_pass(self):
        """Test forward pass with valid input."""
        batch_size = 4
        seq_len = 32
        
        dummy_input = torch.randn(batch_size, seq_len, self.d_model)
        output = self.pos_emb(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))
    
    def test_sequence_too_long_error(self):
        """Test error for sequence longer than max_seq_length."""
        long_input = torch.randn(1, self.max_seq_length + 10, self.d_model)
        
        with self.assertRaises(ValueError):
            self.pos_emb(long_input)
    
    def test_deterministic_output(self):
        """Test that output is deterministic across batches."""
        batch_size = 3
        seq_len = 32
        
        dummy_input = torch.randn(batch_size, seq_len, self.d_model)
        output = self.pos_emb(dummy_input)
        
        # All batch items should have same positional embeddings
        self.assertTrue(torch.allclose(output[0], output[1]))
        self.assertTrue(torch.allclose(output[1], output[2]))
    
    def test_dropout(self):
        """Test dropout functionality."""
        pos_emb_dropout = PositionalEmbedding(self.max_seq_length, self.d_model, dropout=0.5)
        
        # In training mode, dropout should be active
        pos_emb_dropout.train()
        dummy_input = torch.randn(2, 32, self.d_model)
        
        output1 = pos_emb_dropout(dummy_input)
        output2 = pos_emb_dropout(dummy_input)
        
        # With dropout, outputs should be different in training mode
        self.assertFalse(torch.allclose(output1, output2))
        
        # In eval mode, outputs should be same
        pos_emb_dropout.eval()
        output3 = pos_emb_dropout(dummy_input)
        output4 = pos_emb_dropout(dummy_input)
        
        self.assertTrue(torch.allclose(output3, output4))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_causal_mask(self):
        """Test causal mask creation."""
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        # Check shape
        self.assertEqual(mask.shape, (seq_len, seq_len))
        
        # Check that it's upper triangular (True values above diagonal)
        expected = torch.tensor([
            [False, True,  True,  True,  True],
            [False, False, True,  True,  True],
            [False, False, False, True,  True],
            [False, False, False, False, True],
            [False, False, False, False, False]
        ])
        
        self.assertTrue(torch.equal(mask, expected))
    
    def test_create_padding_mask(self):
        """Test padding mask creation."""
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        mask = create_padding_mask(input_ids, pad_token_id=0)
        
        expected = torch.tensor([
            [False, False, False, True, True],
            [False, False, True, True, True]
        ])
        
        self.assertTrue(torch.equal(mask, expected))
    
    def test_count_parameters(self):
        """Test parameter counting."""
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),  # 10*20 + 20 = 220 params
            torch.nn.Linear(20, 5)    # 20*5 + 5 = 105 params
        )
        
        total_params = count_parameters(model, trainable_only=False)
        trainable_params = count_parameters(model, trainable_only=True)
        
        self.assertEqual(total_params, 325)  # 220 + 105
        self.assertEqual(trainable_params, 325)  # All trainable by default
        
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False
        
        trainable_params_frozen = count_parameters(model, trainable_only=True)
        self.assertEqual(trainable_params_frozen, 105)  # Only second layer
    
    def test_save_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        # Create simple model and optimizer
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=10,
                loss=0.5,
                filepath=checkpoint_path,
                custom_data="test"
            )
            
            # Check file exists
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Create new model and optimizer
            new_model = torch.nn.Linear(10, 5)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            
            # Load checkpoint
            checkpoint = load_checkpoint(
                filepath=checkpoint_path,
                model=new_model,
                optimizer=new_optimizer
            )
            
            # Check loaded data
            self.assertEqual(checkpoint['epoch'], 10)
            self.assertEqual(checkpoint['loss'], 0.5)
            self.assertEqual(checkpoint['custom_data'], "test")
            
            # Check model weights are loaded correctly
            for orig_param, loaded_param in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(orig_param, loaded_param))
    
    def test_average_meter(self):
        """Test AverageMeter functionality."""
        meter = AverageMeter('Test', ':.2f')
        
        # Test initial state
        self.assertEqual(meter.avg, 0)
        self.assertEqual(meter.count, 0)
        
        # Add some values
        meter.update(1.0, 2)  # value=1.0, count=2
        self.assertEqual(meter.sum, 2.0)
        self.assertEqual(meter.count, 2)
        self.assertEqual(meter.avg, 1.0)
        
        meter.update(2.0, 3)  # value=2.0, count=3
        self.assertEqual(meter.sum, 8.0)  # 2.0 + 6.0
        self.assertEqual(meter.count, 5)  # 2 + 3
        self.assertEqual(meter.avg, 1.6)  # 8.0 / 5
        
        # Test reset
        meter.reset()
        self.assertEqual(meter.avg, 0)
        self.assertEqual(meter.count, 0)
    
    def test_truncate_or_pad_sequence(self):
        """Test sequence truncation and padding."""
        # Test list input
        sequence = [1, 2, 3]
        
        # Test padding
        padded = truncate_or_pad_sequence(sequence, 5, pad_value=0)
        self.assertEqual(padded, [1, 2, 3, 0, 0])
        
        # Test truncation
        truncated = truncate_or_pad_sequence(sequence, 2)
        self.assertEqual(truncated, [1, 2])
        
        # Test no change
        unchanged = truncate_or_pad_sequence(sequence, 3)
        self.assertEqual(unchanged, [1, 2, 3])
        
        # Test tensor input
        tensor_seq = torch.tensor([1, 2, 3, 4])
        
        # Test padding
        padded_tensor = truncate_or_pad_sequence(tensor_seq, 6, pad_value=-1)
        expected_tensor = torch.tensor([1, 2, 3, 4, -1, -1])
        self.assertTrue(torch.equal(padded_tensor, expected_tensor))
        
        # Test truncation
        truncated_tensor = truncate_or_pad_sequence(tensor_seq, 2)
        expected_truncated = torch.tensor([1, 2])
        self.assertTrue(torch.equal(truncated_tensor, expected_truncated))
    
    def test_set_random_seed(self):
        """Test random seed setting."""
        # Set seed and generate some random numbers
        set_random_seed(42)
        torch_rand1 = torch.rand(3)
        
        # Set different seed
        set_random_seed(123)
        torch_rand2 = torch.rand(3)
        
        # Set original seed again
        set_random_seed(42)
        torch_rand3 = torch.rand(3)
        
        # First and third should be same, different from second
        self.assertTrue(torch.allclose(torch_rand1, torch_rand3))
        self.assertFalse(torch.allclose(torch_rand1, torch_rand2))


class TestDataStructures(unittest.TestCase):
    """Test data-related functionality."""
    
    def test_config_loading(self):
        """Test configuration file structure."""
        # Create a sample config
        sample_config = {
            "model": {
                "max_seq_length": 512,
                "dropout_rate": 0.1,
                "hidden_size": 768
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 1e-5
            }
        }
        
        # Save and load config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config, f)
            config_path = f.name
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Check loaded config
            self.assertEqual(loaded_config["model"]["max_seq_length"], 512)
            self.assertEqual(loaded_config["training"]["batch_size"], 16)
            
        finally:
            os.unlink(config_path)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def test_model_component_compatibility(self):
        """Test that model components work together."""
        batch_size = 2
        seq_len = 32
        d_model = 64
        vocab_size = 1000
        
        # Create components
        pos_emb = PositionalEmbedding(128, d_model)
        embedding = torch.nn.Embedding(vocab_size, d_model)
        
        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass through components
        token_embeddings = embedding(input_ids)
        pos_embeddings = pos_emb(token_embeddings)
        
        # Combined embeddings
        combined = token_embeddings + pos_embeddings
        
        # Check shapes
        self.assertEqual(token_embeddings.shape, (batch_size, seq_len, d_model))
        self.assertEqual(pos_embeddings.shape, (batch_size, seq_len, d_model))
        self.assertEqual(combined.shape, (batch_size, seq_len, d_model))


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestPositionalEmbedding,
        TestUtilityFunctions,
        TestDataStructures,
        TestModelIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)