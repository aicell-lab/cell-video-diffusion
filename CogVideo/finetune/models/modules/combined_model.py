import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union

class CombinedTransformerWithEmbedder(nn.Module):
    """
    Wrapper class that combines the transformer model with the phenotype embedder.
    This avoids DeepSpeed's limitation with multiple models.
    """
    def __init__(self, transformer, phenotype_embedder=None, phenotype_module="single"):
        super().__init__()
        self.transformer = transformer
        self.phenotype_embedder = phenotype_embedder
        self.phenotype_module = phenotype_module
        
        # Forward important attributes from the transformer
        self.config = transformer.config
        
    @property
    def dtype(self):
        """Return the dtype of the wrapped transformer model"""
        transformer_dtype = self.transformer.dtype
        
        # Check for dtype mismatch if phenotype embedder exists
        if self.phenotype_embedder is not None:
            # Get a representative parameter from each
            transformer_param = next(iter(self.transformer.parameters()))
            phenotype_param = next(iter(self.phenotype_embedder.parameters()))
            
            if transformer_param.dtype != phenotype_param.dtype:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Dtype mismatch between transformer ({transformer_param.dtype}) and "
                    f"phenotype embedder ({phenotype_param.dtype}). Using transformer dtype."
                )
        
        return transformer_dtype
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        phenotypes: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        # Process phenotypes if provided and embedder exists
        if phenotypes is not None and self.phenotype_embedder is not None:
            phenotype_embedding = self.phenotype_embedder(phenotypes)
            
            # Handle different phenotype embedding approaches
            if self.phenotype_module == "single":
                # Single token case: discard last token from text embeddings
                encoder_hidden_states = torch.cat(
                    [phenotype_embedding, encoder_hidden_states[:, :-1, :]], 
                    dim=1
                )
            else:  # "multi"
                # Multi-token case: discard last N tokens from text embeddings
                tokens_to_discard = phenotype_embedding.size(1)  # Should be 4
                encoder_hidden_states = torch.cat(
                    [phenotype_embedding, encoder_hidden_states[:, :-tokens_to_discard, :]], 
                    dim=1
                )
        
        # Forward to the transformer with all original arguments
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            timestep_cond=timestep_cond,
            ofs=ofs,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict
        ) 

    def to(self, *args, **kwargs):
        """Handle moving both components to the target device/dtype"""
        self.transformer = self.transformer.to(*args, **kwargs)
        if self.phenotype_embedder is not None:
            self.phenotype_embedder = self.phenotype_embedder.to(*args, **kwargs)
        return self 