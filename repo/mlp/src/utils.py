import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE

def apply_smote_sampling(X_train, y_train, sampling_strategy=0.3):
    """SMOTE를 사용한 오버샘플링"""
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"리샘플링 전: {len(y_train)} → 리샘플링 후: {len(y_resampled)}")
    print(f"사기 비율: {y_train.mean():.3f} → {y_resampled.mean():.3f}")
    return X_resampled, y_resampled


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)