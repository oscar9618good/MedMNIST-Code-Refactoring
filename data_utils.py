import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
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
        raise ValueError(f"不支援的資料集 '{data_flag}' 或該資料集非 28x28。支援的 28x28 資料集有: {SUPPORTED_2D_DATASETS}")

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # 根據通道數決定正規化轉換
    if info['n_channels'] == 3:
        normalize_transform = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    else: # info['n_channels'] == 1
        normalize_transform = transforms.Normalize(mean=[.5], std=[.5])

    # 設定資料轉換流程
    if data_aug and split == 'train':
        transform_list = [
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
        ]
        if info['n_channels'] == 3: # ColorJitter 只適用於彩色圖片
            transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)) # 稍微增加增強選項
        
        transform_list.extend([
            transforms.ToTensor(),
            normalize_transform
        ])
        data_transform = transforms.Compose(transform_list)
    else: # for 'val', 'test' or if data_aug is False
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize_transform
        ])

    dataset = DataClass(split=split, transform=data_transform, download=download)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'), # 訓練時打亂，驗證/測試時不打亂
        num_workers=num_workers,
        pin_memory=True # 如果使用 GPU，建議設定為 True 以加速資料轉移
    )
    return loader, info

if __name__ == '__main__':
    # 測試 data_utils.py
    print("測試 data_utils.py...")
    try:
        print("\n測試 PneumoniaMNIST (灰階)...")
        train_loader_pneumonia, info_pneumonia = get_data_loader('pneumoniamnist', 64, 'train', data_aug=True)
        print(f"PneumoniaMNIST 訓練資料載入成功。Info: {info_pneumonia}")
        images, labels = next(iter(train_loader_pneumonia))
        print(f"一個批次的圖像形狀: {images.shape}, 標籤形狀: {labels.shape}")
        print(f"圖像數據類型: {images.dtype}, 標籤數據類型: {labels.dtype}")
        print(f"圖像值域 (抽樣一個圖像): min={images[0].min()}, max={images[0].max()}")


        print("\n測試 PathMNIST (彩色)...")
        train_loader_path, info_path = get_data_loader('pathmnist', 64, 'train', data_aug=True)
        print(f"PathMNIST 訓練資料載入成功。Info: {info_path}")
        images, labels = next(iter(train_loader_path))
        print(f"一個批次的圖像形狀: {images.shape}, 標籤形狀: {labels.shape}")
        print(f"圖像數據類型: {images.dtype}, 標籤數據類型: {labels.dtype}")
        print(f"圖像值域 (抽樣一個圖像): min={images[0].min()}, max={images[0].max()}")

        
        print("\n測試 RetinaMNIST (彩色, ordinal-regression treated as multi-class)...")
        # MedMNIST 中的 retinamnist 已經是 3x28x28
        test_loader_retina, info_retina = get_data_loader('retinamnist', 64, 'test')
        print(f"RetinaMNIST 測試資料載入成功。Info: {info_retina}")
        images, labels = next(iter(test_loader_retina))
        print(f"一個批次的圖像形狀: {images.shape}, 標籤形狀: {labels.shape}")

        print("\n測試 ChestMNIST (灰階, multi-label)...")
        test_loader_chest, info_chest = get_data_loader('chestmnist', 64, 'test', data_aug=False)
        print(f"ChestMNIST 測試資料載入成功。Info: {info_chest}")
        images, labels = next(iter(test_loader_chest))
        print(f"一個批次的圖像形狀: {images.shape}, 標籤形狀: {labels.shape}") # 標籤應為 (batch_size, num_labels)


    except ValueError as e:
        print(f"錯誤: {e}")
    except Exception as e:
        import traceback
        print(f"發生未預期的錯誤: {e}")
        traceback.print_exc()
    
    print("\ndata_utils.py 測試執行完畢。")