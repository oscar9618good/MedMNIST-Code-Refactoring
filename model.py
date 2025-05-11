import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 輸入圖片大小為 28x28
        # 經過三次 MaxPool2d (kernel_size=2, stride=2)
        # 28 -> 14 -> 7 -> 3 (近似，若 padding 使其保持偶數，則 28->14->7->3.5 -> 向下取整 3)
        # 實際為 28/2 = 14, 14/2 = 7, 7/2 = 3 (取整)
        # 所以 fc 層的輸入大小是 64 * 3 * 3
        self.fc = nn.Linear(64 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # 展平
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # 測試 model.py
    print("測試 model.py...")
    # 假設 PneumoniaMNIST (單通道, 二元分類，但 num_classes 通常設為1用於BCEWithLogitsLoss，或2用於CrossEntropyLoss)
    # MedMNIST 通常將二元分類的 num_classes 設為 len(info['label'])，例如 PneumoniaMNIST 是 ['normal', 'pneumonia'] -> 2
    # 但在實際使用 BCEWithLogitsLoss 時，輸出節點是 1。
    # 為了與 MedMNIST evaluator 兼容，通常使用 CrossEntropyLoss，所以 num_classes 應為實際類別數。
    
    # 單通道, 2 個類別 (例如 PneumoniaMNIST)
    model_pneumonia = SimpleCNN(in_channels=1, num_classes=2)
    dummy_input_pneumonia = torch.randn(64, 1, 28, 28) # batch_size, channels, height, width
    output_pneumonia = model_pneumonia(dummy_input_pneumonia)
    print(f"PneumoniaMNIST 模型輸出形狀: {output_pneumonia.shape}") # 應為 (64, 2)

    # 三通道, 假設有 9 個類別 (例如 PathMNIST)
    model_pathmnist = SimpleCNN(in_channels=3, num_classes=9)
    dummy_input_pathmnist = torch.randn(64, 3, 28, 28)
    output_pathmnist = model_pathmnist(dummy_input_pathmnist)
    print(f"PathMNIST 模型輸出形狀: {output_pathmnist.shape}") # 應為 (64, 9)