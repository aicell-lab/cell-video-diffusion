import torch
import torch.nn as nn

class PhenotypeEmbedder(nn.Module):
    """
    Module that embeds phenotype vectors into the same dimension as text embeddings.
    
    Args:
        input_dim (int): Dimension of input phenotype vectors (default: 4)
        hidden_dim (int): Dimension of hidden layer (default: 256)
        output_dim (int): Dimension of output embeddings (default: 4096)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=4096, dropout=0.1):
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

class PhenotypeEmbedderMulti(nn.Module):
    """
    Module that embeds each phenotype dimension separately into tokens of the same dimension 
    as text embeddings.
    
    Args:
        input_dim (int): Dimension of input phenotype vectors (default: 4)
        hidden_dim (int): Dimension of hidden layer for each embedder (default: 256)
        output_dim (int): Dimension of output embeddings (default: 4096)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=4096, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create separate embedders for each phenotype dimension
        self.embedders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, output_dim),
            ) for _ in range(input_dim)
        ])
        
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
            torch.Tensor: Embedded phenotypes of shape [batch_size, input_dim, output_dim]
                          where each phenotype dimension gets its own token
        """
        batch_size = phenotypes.size(0)
        
        # Process each phenotype dimension separately
        embeddings = []
        for i in range(self.input_dim):
            # Extract single dimension and keep dim: [batch_size, 1]
            phenotype_dim = phenotypes[:, i:i+1]
            
            # Embed this dimension: [batch_size, output_dim]
            embedded_dim = self.embedders[i](phenotype_dim)
            
            embeddings.append(embedded_dim)
        
        # Stack along sequence dimension: [batch_size, input_dim, output_dim]
        stacked_embeddings = torch.stack(embeddings, dim=1)
        
        return stacked_embeddings 