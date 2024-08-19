import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, d_model):
        super(PositionalEmbedding, self).__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model

        # Create sinusoidal positional embeddings
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_embedding', pe.unsqueeze(0))  # Shape: [1, max_seq_length, d_model]

    def forward(self, x):
        return self.positional_embedding[:, :x.size(1), :].to(x.device)

def plot_positional_embeddings(positional_embedding):
    pe = positional_embedding.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU for plotting
    plt.figure(figsize=(10, 6))
    plt.imshow(pe, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Positional Embeddings')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position in Sequence')
    plt.show()

if __name__ == "__main__":
    max_seq_length = 50  # Example sequence length
    d_model = 16  # Example embedding dimension
    
    positional_embedding_layer = PositionalEmbedding(max_seq_length, d_model)
    
    # Generate positional embeddings
    dummy_input = torch.zeros(1, max_seq_length, d_model)  # Batch size of 1
    positional_embeddings = positional_embedding_layer(dummy_input)
    
    # Plot the positional embeddings
    plot_positional_embeddings(positional_embeddings)
    
    positional_embedding_layer = PositionalEmbedding(max_seq_length, d_model)
    
    # Generate positional embeddings
    dummy_input = torch.zeros(1, max_seq_length, d_model)  # Batch size of 1
    positional_embeddings = positional_embedding_layer(dummy_input)
    
    # Print the positional embeddings
    print("Positional Embeddings:")
    print(positional_embeddings.squeeze(0).cpu().numpy())

    # Check if all values are zeros
    are_all_zeros = torch.all(positional_embeddings == 0).item()
    print(f"Are all values zeros? {are_all_zeros}")
