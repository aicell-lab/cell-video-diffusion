import torch
import torch.nn as nn

class PhenotypeEmbedder(nn.Module):
    """
    Module that embeds phenotype vectors into the same dimension as text embeddings.
    
    Args:
        input_dim (int): Dimension of input phenotype vectors (default: 3)
        hidden_dim (int): Dimension of hidden layer (default: 256)
        output_dim (int): Dimension of output embeddings (default: 4096)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=4096, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Multi-layer MLP to transform phenotype vectors to text embedding dimension
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, phenotypes):
        """
        Args:
            phenotypes (torch.Tensor): Phenotype vectors of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Embedded phenotypes of shape [batch_size, 1, output_dim]
        """
        # [batch_size, input_dim] -> [batch_size, output_dim]
        embeddings = self.layers(phenotypes)
        
        # Add sequence dimension: [batch_size, output_dim] -> [batch_size, 1, output_dim]
        embeddings = embeddings.unsqueeze(1)
        
        return embeddings 