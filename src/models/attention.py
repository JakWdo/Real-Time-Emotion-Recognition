# src/models/attention.py

import torch.nn as nn

class AttentionModule(nn.Module):
    """
    Moduł mechanizmu uwagi do podkreślania istotnych cech w mapach aktywacji.
    Implementuje mechanizm Squeeze-and-Excitation.
    """
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            # Redukcja wymiarowości (squeeze)
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            # Przywrócenie oryginalnego rozmiaru (excitation)
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()  # Normalizacja wag uwagi do zakresu [0,1]
        )

    def forward(self, x):
        """
        Aplikuje mechanizm uwagi na wejściowych cechach.
        
        Args:
            x: tensor wejściowy [batch_size, channels, height, width]
        Returns:
            tensor ważony uwagą tego samego rozmiaru co wejście
        """
        attention_weights = self.attention(x)
        return x * attention_weights  # Ważenie elementowe
