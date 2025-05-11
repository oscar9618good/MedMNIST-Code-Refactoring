# MedMNIST 2D (28x28) 影像分類專案

一個基於 **PyTorch** 的模組化專案，用於訓練與評估 **MedMNIST** 2D (28x28) 醫療影像資料集的分類模型。

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.7%2B-orange.svg)](https://pytorch.org/)  
[![MedMNIST](https://img.shields.io/badge/dataset-MedMNIST-green.svg)](https://medmnist.com/)

---

##  目錄

- [專案概覽](#專案概覽)  
- [專案動機](#專案動機)  
- [主要功能](#主要功能)  
- [技術棧](#技術棧)  
- [支援的資料集](#支援的資料集)  
- [專案結構](#專案結構)  
- [環境要求與安裝](#環境要求與安裝)  
- [如何使用](#如何使用)  
  - [訓練模型](#訓練模型)  
  - [測試模型](#測試模型)  
- [模型架構](#模型架構)  
- [基礎測試](#基礎測試)  
- [如何貢獻](#如何貢獻)  
- [授權條款](#授權條款)  
- [致謝](#致謝)  
- [未來展望](#未來展望)

---

##  專案概覽

本專案提供一個模組化的訓練框架，幫助使用者快速在 MedMNIST 2D (28x28) 資料集上訓練、測試影像分類模型，支援擴充與修改，適合教學與研究用途。

##  專案動機

- 降低初學者上手 PyTorch 與 MedMNIST 的門檻  
- 提供結構清晰的程式碼，利於教學與再利用  
- 支援多種醫療影像分類任務，促進快速實驗與比較

##  主要功能

- 模組化架構：清楚區分資料處理、模型定義、訓練與測試流程  
- 支援多個 MedMNIST 資料集  
- CLI 介面：使用 `argparse` 提供彈性設定  
- 資料增強選項  
- 模型儲存與載入功能  
- 使用 `medmnist.Evaluator` 進行標準化評估  

##  技術棧

- Python 3.7+  
- PyTorch 1.7+  
- torchvision  
- medmnist  
- numpy  
- tqdm  

##  支援的資料集

- `pathmnist`  
- `chestmnist`  
- `dermamnist`  
- `octmnist`  
- `pneumoniamnist`  
- `retinamnist`  
- `breastmnist`  
- `bloodmnist`  
- `tissuemnist`  
- `organamnist`  
- `organcmnist`  
- `organsmnist`

##  專案結構

```
medmnist_project/
├── data_utils.py      # 載入與預處理資料
├── model.py           # 模型定義 (SimpleCNN)
├── train.py           # 模型訓練
├── test.py            # 模型測試
├── utils.py           # 儲存/載入模型等工具
└── README.md

+-----------------+      +-----------------+      +-----------------+
|   train.py      |----->|  data_utils.py  |<-----|    test.py      |
| (主腳本)        |      | get_data_loader |      | (主腳本)         |
+-----------------+      +-----------------+      +-----------------+
        |                        |                        |
        |                        |                        |
        v                        v                        v
+-----------------+      +-----------------+      +-----------------+
|    model.py     |<-----|                 |<-----|    utils.py     |
|   SimpleCNN     |      | (PyTorch,       |      | save_model      |
+-----------------+      |  MedMNIST lib)  |      | load_model      |
                         +-----------------+      +-----------------+
        ^                                                 ^
        |                                                 |
        +-------------------------------------------------+

```
---

##  環境要求與安裝

###  先決條件

- Python 3.7 以上

###  安裝步驟

1. 複製專案（若使用 Git）：
   ```bash
   git clone <your-repository-url>
   cd medmnist_project
   ```

2. 安裝必要套件：
   ```bash
   pip install torch torchvision medmnist numpy tqdm
   ```

---

##  如何使用

###  訓練模型

```bash
python train.py --data_flag <dataset_name> [其他選項]
```

#### 範例 1：
```bash
python train.py --data_flag pneumoniamnist --num_epochs 10 --lr 0.001 --batch_size 128 --data_aug
```

#### 範例 2：
```bash
python train.py --data_flag pathmnist --num_epochs 15 --output_root ./my_custom_models --model_filename pathmnist_best.pth
```

#### 參數說明：

| 參數           | 說明                                       | 預設值             |
|----------------|-------------------------------------------|--------------------|
| `--data_flag`  | 使用的資料集名稱（必要）                     | 無                 |
| `--output_root`| 模型儲存根目錄                              | `./trained_models` |
| `--num_epochs` | 訓練週期數                                 | `10`               |
| `--lr`         | 學習率                                     | `0.001`            |
| `--batch_size` | 批次大小                                   | `128`              |
| `--num_workers`| DataLoader 使用的工作執行緒                 | `4`                |
| `--data_aug`   | 是否使用資料增強                            | `False`            |
| `--model_filename`| 模型檔案名稱                             | `model.pth`        |

---

###  測試模型

```bash
python test.py --data_flag <dataset_name> --model_path_root <path_to_models_dir> [其他選項]
```

#### 範例 1：
```bash
python test.py --data_flag pneumoniamnist --model_path_root ./trained_models --model_filename model.pth
```

#### 範例 2：
```bash
python test.py --data_flag pathmnist --model_path_root ./my_custom_models --model_filename pathmnist_best.pth
```

---

##  模型架構 

```
(SimpleCNN)
輸入: (B, C, 28, 28)
↓ Conv2d(16 filters) + BN + ReLU + MaxPool(2x2)
↓ Conv2d(32 filters) + BN + ReLU + MaxPool(2x2)
↓ Conv2d(64 filters) + BN + ReLU + MaxPool(2x2)
↓ Flatten
↓ Fully Connected → 輸出類別數
```

---

##  基礎測試

你可以執行下列指令來驗證各模組是否正常工作：

```bash
python data_utils.py
python model.py
python utils.py
```

---

##  如何貢獻

1. Fork 專案  
2. 建立分支：`git checkout -b feature/AmazingFeature`  
3. 提交修改：`git commit -m 'Add some AmazingFeature'`  
4. 推送：`git push origin feature/AmazingFeature`  
5. 發送 Pull Request！

請遵守程式碼風格，並確認變更不會破壞現有功能。

---

##  授權條款

本專案採用 **MIT License**。請參考專案中的 `LICENSE.md` 檔案。

---

##  致謝

- 感謝 MedMNIST 團隊提供標準化醫療影像資料集  
- 感謝 PyTorch 社群的持續貢獻

---

##  未來展望

-  支援更多 MedMNIST 資料集（如 3D）  
-  整合更進階的模型架構  
-  加入單元測試（如 `pytest`）  
-  整合 TensorBoard 視覺化工具  
-  提供 `requirements.txt` 以利環境重建
