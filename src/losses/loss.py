import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', dynamic_alpha=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.dynamic_alpha = dynamic_alpha

    def forward(self, preds, targets):
        """
        Args:
            preds (Tensor): Logits [batch_size, num_classes]
            targets (Tensor): Class indices [batch_size]
        """
        probs = torch.softmax(preds, dim=1)
        pt = probs[torch.arange(targets.size(0)), targets]
        ce_loss = -torch.log(pt + 1e-8)
        
        modulating_factor = (1 - pt) ** self.gamma
        alpha = self.alpha if not self.dynamic_alpha else (1 - pt)
        
        focal_loss = alpha * modulating_factor * ce_loss
        
        return focal_loss.mean() if self.reduction == 'mean' else (
            focal_loss.sum() if self.reduction == 'sum' else focal_loss
        )

class CKALoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def centering(self, K):
        """
        Centers the kernel matrix K using the centering matrix H = I - (1/n)11^T
        Args:
            K: Kernel matrix of shape [n x n]
        Returns:
            Centered kernel matrix
        """
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - (1.0/n) * torch.ones((n, n), device=K.device)
        return H @ K @ H
    
    def forward(self, wav_features, rob_features):
        """
        Compute CKA loss between WavLM and RoBERTa features
        Args:
            wav_features: WavLM features after transformer [batch_size x hidden_dim]
            rob_features: RoBERTa features after transformer [batch_size x hidden_dim]
        Returns:
            CKA loss (1 - CKA since we want to maximize alignment)
        """
        # Compute Gram matrices
        K = wav_features @ wav_features.T  # [batch_size x batch_size]
        L = rob_features @ rob_features.T  # [batch_size x batch_size]
        
        # Center Gram matrices
        K_centered = self.centering(K)
        L_centered = self.centering(L)
        
        # Compute HSIC
        HSIC_KL = torch.trace(K_centered @ L_centered)
        HSIC_KK = torch.trace(K_centered @ K_centered)
        HSIC_LL = torch.trace(L_centered @ L_centered)
        
        # Compute CKA
        epsilon = 1e-8  # Small constant for numerical stability
        CKA = HSIC_KL / (torch.sqrt(HSIC_KK * HSIC_LL) + epsilon)
        
        # Return loss (1 - CKA since we want to maximize alignment)
        return 1 - CKA

def differentiable_f1_loss(y_pred, y_true, epsilon=1e-7):
    """
    Calculate differentiable F1 score loss.
    
    Args:
        y_pred: Predicted probabilities (batch_size, num_classes)
        y_true: One-hot encoded ground truth (batch_size, num_classes)
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Differentiable F1 loss (to be minimized)
    """
    # Apply sigmoid if predictions are logits
    y_pred = torch.sigmoid(y_pred)
    
    # Calculate true positives, false positives, false negatives
    tp = torch.sum(y_pred * y_true, dim=0)
    fp = torch.sum(y_pred * (1 - y_true), dim=0)
    fn = torch.sum((1 - y_pred) * y_true, dim=0)
    
    # Calculate precision and recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # Average across classes (macro averaging)
    macro_f1 = torch.mean(f1)
    
    # Return 1 - F1 score as the loss to minimize
    return 1 - macro_f1

# Optional wrapper to ensure even more stability
class DiffF1Loss(torch.nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        return differentiable_f1_loss(y_pred, y_true, self.epsilon)

class HierarchicalLoss(nn.Module):
    def __init__(self, class_weights=None, similarity_weight=0.1):
        """
        Args:
            emotion_similarity_matrix: Matrix of shape [num_emotions, num_emotions] defining emotion similarities
            class_weights: Optional tensor of weights for each class for weighted CE loss
            similarity_weight: Weight for the similarity-based loss component (default 0.1)
        """
        super().__init__()
        self.class_weights = class_weights
        self.similarity_weight = similarity_weight
        
        # Example usage of similarity matrix for 8 emotions:
        self.similarity_matrix = torch.tensor([
            # Angry  Sad    Happy  Surpr  Fear   Disg   Cont   Neut
            [1.00,  0.30,  0.10,  0.25,  0.30,  0.60,  0.70,  0.20],  # Angry
            [0.30,  1.00,  0.10,  0.20,  0.40,  0.30,  0.40,  0.50],  # Sad
            [0.10,  0.10,  1.00,  0.60,  0.15,  0.10,  0.15,  0.40],  # Happy
            [0.25,  0.20,  0.60,  1.00,  0.50,  0.20,  0.20,  0.30],  # Surprise
            [0.30,  0.40,  0.15,  0.50,  1.00,  0.40,  0.30,  0.25],  # Fear
            [0.60,  0.30,  0.10,  0.20,  0.40,  1.00,  0.65,  0.25],  # Disgust
            [0.70,  0.40,  0.15,  0.20,  0.30,  0.65,  1.00,  0.35],  # Contempt
            [0.20,  0.50,  0.40,  0.30,  0.25,  0.25,  0.35,  1.00],  # Neutral
        ]).cuda()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        # Weighted cross-entropy loss
        ce_loss = F.cross_entropy(
            predictions, 
            targets,
            weight=self.class_weights  # None if no class weights provided
        )
        
        # Convert targets to one-hot and apply similarity weighting
        soft_targets = F.one_hot(targets, num_classes=predictions.size(-1)).float()
        soft_targets = torch.matmul(soft_targets, self.similarity_matrix)
        
        # Normalize soft targets
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        
        # KL divergence with similarity-weighted targets
        kl_loss = F.kl_div(
            F.log_softmax(predictions, dim=-1),
            soft_targets,
            reduction='batchmean'
        )
        
        # Combine losses
        total_loss = ce_loss + self.similarity_weight * kl_loss
        
        return total_loss



class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Cross Entropy Loss with label smoothing and class weights.
        
        Args:
            smoothing: Label smoothing factor (0-1)
            class_weights: Tensor of shape (num_classes,) for class weights
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the loss function.
        
        Args:
            logits: Raw predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Loss value
        """
        num_classes = logits.size(-1)
        
        # Create one-hot vectors for targets
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        if self.smoothing > 0:
            one_hot = (1 - self.smoothing) * one_hot + \
                     self.smoothing / num_classes
        
        # Compute cross entropy
        log_probs = torch.log_softmax(logits, dim=-1)
        losses = -(one_hot * log_probs)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            losses = losses * self.class_weights.unsqueeze(0)
        
        # Sum across classes
        losses = losses.sum(dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        return losses