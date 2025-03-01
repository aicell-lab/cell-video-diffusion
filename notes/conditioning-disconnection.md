Scientific plan for fixing the conditioning disconnection problem

### First Frame Emphasis Loss

**Problem Statement:**
During training, we observe a disconnection between the conditioning frame (frame0) and the first generated frame (frame1). While early in training the model maintains continuity between these frames, later in training frame1 becomes significantly darker with barely visible cells, essentially "resetting" before generating a proper cell division sequence in later frames. This breaks the continuity from the conditioning frame.

**Proposed Solution:**
Implement a "First Frame Emphasis Loss" that specifically penalizes discrepancies between the predicted first frame and its ground truth. This directly addresses the conditioning disconnection by adding a weighted term to the loss function that focuses on the critical transition between conditioning and generation.

**Implementation:**
```python
# Extract first frame predictions and ground truth
first_frame_pred = latent_pred[:, 0]  # First predicted frame
first_frame_gt = latent[:, 0]         # First ground truth frame
    
# Calculate MSE specifically for the first frame
first_frame_loss = torch.mean((first_frame_pred - first_frame_gt) ** 2, dim=(1, 2, 3)).mean()
    
# Add to original loss with a higher weight
total_loss = original_loss + first_frame_weight * first_frame_loss
```

**Experimental Plan:**
1. Fix LoRA rank at 128 for all experiments to ensure fair comparison
2. Train multiple models with different values for the first_frame_weight parameter:
   - Weight = 0 (baseline/control)
   - Weight = 2.0 (mild emphasis)
   - Weight = 5.0 (medium emphasis)
   - Weight = 10.0 (strong emphasis)

3. Evaluation metrics:
   - Visual assessment of frame0→frame1 continuity
   - SSIM between consecutive frames (especially frame0→frame1)
   - Cell detection and tracking metrics if available
   - Overall video generation quality

**Expected Outcome:**
By placing additional emphasis on the first frame, we expect the model to maintain stronger continuity from the conditioning frame, ensuring that cells present in frame0 properly persist into frame1 and continue their division process in subsequent frames, rather than "starting over" with a dark frame.

**Potential Concerns:**
- Too high a weight might sacrifice quality in later frames
- Need to balance first-frame accuracy with overall sequence quality
- May require adjusting learning rate due to potentially larger gradients