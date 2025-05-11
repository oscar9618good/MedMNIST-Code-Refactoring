import argparse
import torch
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import roc_auc_score, accuracy_score 

from data_utils import get_data_loader, SUPPORTED_2D_DATASETS
from model import SimpleCNN
from utils import load_model


def test_model(model, test_loader, device, task_type):
    model.eval()
    # y_true_test 和 y_score_test 在迴圈內部處理
    all_labels_list = []
    all_scores_list = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="測試中"):
            inputs_on_device = inputs.to(device) # labels 不立即移到 device，因為我們需要原始的 labels for all_labels_list
            
            outputs = model(inputs_on_device) # logits

            # 根據任務類型獲取用於指標計算的分數
            if task_type == 'multi-label, binary-class':
                scores_for_metric = torch.sigmoid(outputs).detach()
            else: # binary-class 和 multi-class
                scores_for_metric = outputs.detach()

            all_labels_list.append(labels.cpu()) # 收集原始標籤 (CPU)
            all_scores_list.append(scores_for_metric.cpu()) # 收集模型輸出分數 (CPU)
            
    # 在函數末尾將列表中的 tensors 合併並轉換為 numpy arrays
    final_y_true = torch.cat(all_labels_list, dim=0).numpy()
    final_y_score = torch.cat(all_scores_list, dim=0).numpy()

    return final_y_true, final_y_score

def main():
    parser = argparse.ArgumentParser(description="MedMNIST 2D 資料集測試腳本")
    parser.add_argument('--data_flag', type=str, default='pneumoniamnist',
                        help=f"MedMNIST 資料集標籤 (僅限 28x28)。可選: {', '.join(SUPPORTED_2D_DATASETS)}")
    parser.add_argument('--model_path_root', type=str, default='./trained_models',
                        help="已儲存模型權重的根目錄")
    parser.add_argument('--model_filename', type=str, default="model.pth", help="儲存的模型檔案名稱")
    parser.add_argument('--batch_size', type=int, default=128, help="批次大小")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader 的工作執行緒數量")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    model_full_path_dir = os.path.join(args.model_path_root, args.data_flag)

    print(f"載入測試資料集: {args.data_flag}...")
    test_loader, info = get_data_loader(args.data_flag, args.batch_size, 'test', 
                                        download=True, num_workers=args.num_workers, data_aug=False)
    
    n_channels = info['n_channels']
    task = info['task']

    if task == "multi-label, binary-class":
        n_classes = len(info['label'])
    elif 'multi-class' in task:
        n_classes = len(info['label'])
    elif 'binary-class' in task:
        n_classes = 2
    else:
        raise ValueError(f"未知的任務類型: {task}")

    print(f"資料集資訊: 通道數={n_channels}, 類別數={n_classes}, 任務類型={task}")

    model = SimpleCNN(in_channels=n_channels, num_classes=n_classes) # 初始化模型結構
    
    try:
        model = load_model(model, model_full_path_dir, args.model_filename, device)
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print(f"請確保模型 '{os.path.join(model_full_path_dir, args.model_filename)}' 存在。")
        print("你可以先執行 train.py 來訓練並儲存一個模型。")
        return
    
    
    # 測試模型，獲取真實標籤和模型分數 (numpy arrays)
    test_y_true_np, test_y_score_np = test_model(model, test_loader, device, task)
    
    # --- 開始使用 scikit-learn 計算指標 ---
    auc_value = 0.0
    acc_value = 0.0

    # 1. 準備 y_true 給 scikit-learn (通常是 1D array)
    test_y_true_sklearn = test_y_true_np.squeeze()

    if task == "binary-class":
        # test_y_score_np 是 (N, 2) 的 logits
        y_score_logits_tensor = torch.from_numpy(test_y_score_np) # 不需要 .to(device)
        y_score_probs = torch.softmax(y_score_logits_tensor, dim=1)
        y_score_positive_class_probs_np = y_score_probs[:, 1].numpy()
        try:
            auc_value = roc_auc_score(test_y_true_sklearn, y_score_positive_class_probs_np)
        except ValueError as e:
            print(f"警告：計算 AUC 時出錯 ({e})。可能是所有樣本都屬於同一類別。將 AUC 設為 0.5。")
            auc_value = 0.5
        
        predicted_labels_np = np.argmax(test_y_score_np, axis=1)
        acc_value = accuracy_score(test_y_true_sklearn, predicted_labels_np)
        
        print(f"測試結果 ({args.data_flag}):")
        print(f"  AUC: {auc_value:.4f}")
        print(f"  Accuracy: {acc_value:.4f}")

    elif task == "multi-label, binary-class":
        # test_y_score_np 是 (N, num_labels) 的 sigmoid 機率
        try:
            auc_value = roc_auc_score(test_y_true_sklearn, test_y_score_np, average='macro')
        except ValueError as e:
            print(f"警告：計算多標籤 AUC 時出錯 ({e})。將 AUC 設為 0.5。")
            auc_value = 0.5
        
        predicted_labels_np = (test_y_score_np > 0.5).astype(int)
        acc_value = accuracy_score(test_y_true_sklearn, predicted_labels_np) # Subset accuracy
        
        print(f"測試結果 ({args.data_flag}):")
        print(f"  AUC (Macro): {auc_value:.4f}")
        print(f"  Accuracy (Subset): {acc_value:.4f}") # 註明是子集準確率

    else: # multi-class
        predicted_labels_np = np.argmax(test_y_score_np, axis=1)
        acc_value = accuracy_score(test_y_true_sklearn, predicted_labels_np)
        
        y_score_logits_tensor = torch.from_numpy(test_y_score_np)
        y_score_probs_np = torch.softmax(y_score_logits_tensor, dim=1).numpy()
        try:
            auc_value = roc_auc_score(test_y_true_sklearn, y_score_probs_np, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"警告：計算多類別 AUC 時出錯 ({e})。將 AUC 設為 0.0。")
            auc_value = 0.0
            
        print(f"測試結果 ({args.data_flag}):")
        print(f"  Accuracy: {acc_value:.4f}")
        print(f"  AUC (Macro OvR): {auc_value:.4f}")
    # --- 指標計算結束 ---

    print("測試完成。")

if __name__ == '__main__':
    main()