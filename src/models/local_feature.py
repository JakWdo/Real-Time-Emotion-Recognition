# src/models/local_feature.py

import torch
import torch.nn as nn

class LocalFeatureExtractor(nn.Module):
    """
    Ekstraktor cech lokalnych przetwarzający różne regiony twarzy niezależnie.
    Dzieli obraz na 4 części i przetwarza każdą z nich osobno.
    """
    def __init__(self, inplanes, planes):
        super(LocalFeatureExtractor, self).__init__()
        # Warstwy dla pierwszego regionu (lewy górny)
        self.conv1_1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, 
                                padding=1, groups=inplanes, bias=False)
        self.bn1_1 = nn.BatchNorm2d(inplanes)
        self.conv1_2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(planes)

        # Warstwy dla drugiego regionu (prawy górny)
        self.conv2_1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, 
                                padding=1, groups=inplanes, bias=False)
        self.bn2_1 = nn.BatchNorm2d(inplanes)
        self.conv2_2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(planes)

        # Warstwy dla trzeciego regionu (lewy dolny)
        self.conv3_1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, 
                                padding=1, groups=inplanes, bias=False)
        self.bn3_1 = nn.BatchNorm2d(inplanes)
        self.conv3_2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(planes)

        # Warstwy dla czwartego regionu (prawy dolny)
        self.conv4_1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, 
                                padding=1, groups=inplanes, bias=False)
        self.bn4_1 = nn.BatchNorm2d(inplanes)
        self.conv4_2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Przetwarza obraz dzieląc go na 4 regiony i aplikując niezależne konwolucje.
        
        Args:
            x: tensor wejściowy [batch_size, channels, height, width]
        Returns:
            połączone cechy ze wszystkich regionów
        """
        # Podział na regiony
        patch_11 = x[:, :, 0:28, 0:28]
        patch_21 = x[:, :, 28:56, 0:28]
        patch_12 = x[:, :, 0:28, 28:56]
        patch_22 = x[:, :, 28:56, 28:56]

        # Przetwarzanie każdego regionu
        out_1 = self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(patch_11))))))
        out_2 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(patch_21))))))
        out_3 = self.relu(self.bn3_2(self.conv3_2(self.relu(self.bn3_1(self.conv3_1(patch_12))))))
        out_4 = self.relu(self.bn4_2(self.conv4_2(self.relu(self.bn4_1(self.conv4_1(patch_22))))))

        # Łączenie cech
        out1 = torch.cat([out_1, out_2], dim=2)
        out2 = torch.cat([out_3, out_4], dim=2)
        out = torch.cat([out1, out2], dim=3)

        return out
