import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCustomerAwareEnsemble(nn.Module):
    """고객 정보를 활용한 향상된 앙상블 모델"""

    def __init__(self, customer_dim, card_dim, risk_dim, ensemble_dim=6, hidden_dim=256):
        super().__init__()

        # 고객 정보 처리 네트워크
        self.customer_network = nn.Sequential(
            nn.Linear(customer_dim, hidden_dim),  # 22 -> 256
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )

        # 카드 정보 처리 네트워크
        self.card_network = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),  # 14 -> 256
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )

        # 리스크 정보 처리 네트워크
        self.risk_network = nn.Sequential(
            nn.Linear(risk_dim, hidden_dim // 2),  # 2 -> 128
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )

        # 앙상블 예측 처리
        self.ensemble_network = nn.Sequential(
            nn.Linear(ensemble_dim, hidden_dim // 2),  # 6 -> 128
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )

        # 통합 네트워크
        combined_dim = (hidden_dim // 2) * 4  # 128 * 4 = 512
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_customer, x_card, x_risk, x_ensemble):
        customer_features = self.customer_network(x_customer)
        card_features = self.card_network(x_card)
        risk_features = self.risk_network(x_risk)
        ensemble_features = self.ensemble_network(x_ensemble)

        combined = torch.cat([customer_features, card_features, risk_features, ensemble_features], dim=1)
        output = self.fusion_network(combined)
        return output  # logits 반환