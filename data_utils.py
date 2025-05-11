import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO

SUPPORTED_2D_DATASETS = [
    'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
    'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist',
    'organcmnist', 'organsmnist'
] 

def get_data_loader(data_flag, batch_size, split, download=True, num_workers=4, data_aug=False):
    """
    為指定的 MedMNIST 2D 資料集建立 DataLoader。

    Args:
        data_flag (str): MedMNIST 資料集的名稱 (例如 'pneumoniamnist')。
        batch_size (int): DataLoader 的批次大小。
        split (str): 'train', 'val', 或 'test'。
        download (bool): 如果資料集不存在，是否下載。
        num_workers (int): DataLoader 的工作執行緒數量。
        data_aug (bool): 是否對訓練集套用資料增強。

    Returns:
        torch.utils.data.DataLoader: 設定好的 DataLoader。
        dict: 資料集的資訊 (從 medmnist.INFO)。
    """
    if data_flag.lower() not in SUPPORTED_2D_DATASETS:
        # 檢查 info[data_flag]['n_channels'] 來確認是否為灰階或彩色
        # 並檢查圖片大小是否為 28x28
        # 為了簡化，我們假設所有 SUPPORTED_2D_DATASETS 都是 28x28
        # 實際 MedMNIST 中有些資料集如 OrganMNIST* 是 1x28x28
        # 而有些如 PathMNIST 是 3x28x28
        raise ValueError(f"不支援的資料集 '{data_flag}' 或該資料集非 28x28。支援的 28x28 資料集有: {SUPPORTED_2D_DATASETS}")

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # 預處理轉換
    if data_aug and split == 'train':
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]) # 假設單通道，若多通道需調整
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]) # 假設單通道，若多通道需調整
        ])
    
    # 根據通道數調整正規化
    if info['n_channels'] == 3:
        normalize_transform = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    else: # info['n_channels'] == 1
        normalize_transform = transforms.Normalize(mean=[.5], std=[.5])

    if data_aug and split == 'train':
        transform_list = [
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
        ]
        if info['n_channels'] == 3: # ColorJitter 只適用於彩色圖片
            transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        
        transform_list.extend([
            transforms.ToTensor(),
            normalize_transform
        ])
        data_transform = transforms.Compose(transform_list)
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize_transform
        ])


    dataset = DataClass(split=split, transform=data_transform, download=download)
    
    # 對於像 PneumoniaMNIST 這樣的二元分類任務，info['label'] 可能不直接給出 'n_classes'
    # 我們可以從 task 推斷，或者如果它是多標籤，則 n_classes 是標籤的數量
    # MedMNIST Evaluator 會處理這個
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'), # 訓練時打亂，驗證/測試時不打亂
        num_workers=num_workers
    )
    return loader, info

if __name__ == '__main__':
    # 測試 data_utils.py
    print("測試 data_utils.py...")
    try:
        train_loader_pneumonia, info_pneumonia = get_data_loader('pneumoniamnist', 64, 'train', data_aug=True)
        print(f"PneumoniaMNIST 訓練資料載入成功。Info: {info_pneumonia}")
        print(f"一個批次的資料形狀: {next(iter(train_loader_pneumonia))[0].shape}")

        train_loader_path, info_path = get_data_loader('pathmnist', 64, 'train', data_aug=True)
        print(f"PathMNIST 訓練資料載入成功。Info: {info_path}")
        print(f"一個批次的資料形狀: {next(iter(train_loader_path))[0].shape}")
        
        # 測試不支援的資料集
        # get_data_loader('retinamnist', 64, 'train') # RetinaMNIST 是 3x224x224，應該會報錯（目前是簡化假設）
        # 根據 SUPPORTED_2D_DATASETS，retinamnist 是 28x28
        test_loader_retina, info_retina = get_data_loader('retinamnist', 64, 'test')
        print(f"RetinaMNIST 測試資料載入成功。Info: {info_retina}")
        print(f"一個批次的資料形狀: {next(iter(test_loader_retina))[0].shape}")


    except ValueError as e:
        print(f"錯誤: {e}")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")