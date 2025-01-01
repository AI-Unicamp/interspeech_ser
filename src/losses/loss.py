import torch
import torch.nn as nn

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