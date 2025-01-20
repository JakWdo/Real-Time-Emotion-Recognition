# src/models/efficient_face.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .attention import AttentionModule
from .local_feature import LocalFeatureExtractor

class EfficientFaceResNet(nn.Module):
    """
    Model hybrydowy łączący ResNet z ekstraktorem cech lokalnych i mechanizmami uwagi
    do rozpoznawania emocji w czasie rzeczywistym.
    """
    def __init__(self, num_classes):
        super(EfficientFaceResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.local_feature_extractor = LocalFeatureExtractor(3, 116)
        self.attention1 = AttentionModule(256)
        self.attention2 = AttentionModule(512)
        self.attention3 = AttentionModule(1024)
        self.attention4 = AttentionModule(2048)
        
        # Adapter cech lokalnych do wymiaru cech globalnych
        self.local_feature_adapter = nn.Conv2d(116, 2048, kernel_size=1)
        self.fc_emotion = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Ekstrakcja cech lokalnych
        local_features = self.local_feature_extractor(x)

        # ResNet z mechanizmami uwagi
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.attention1(x)

        x = self.resnet.layer2(x)
        x = self.attention2(x)

        x = self.resnet.layer3(x)
        x = self.attention3(x)

        x = self.resnet.layer4(x)
        x = self.attention4(x)

        # Integracja cech lokalnych i globalnych
        adapted_local_features = self.local_feature_adapter(local_features)
        adapted_local_features = F.interpolate(
            adapted_local_features, 
            size=x.size()[2:], 
            mode='bilinear',
            align_corners=False
        )
        x = x + adapted_local_features

        # Klasyfikacja emocji
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        emotion_output = self.fc_emotion(x)
        emotion_output = F.log_softmax(emotion_output, dim=1)

        return emotion_output
