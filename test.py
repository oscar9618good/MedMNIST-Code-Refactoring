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
    all_labels_list = []
    all_scores_list = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="測試中"):
            inputs_on_device = inputs.to(device)
            outputs = model(inputs_on_device) # logits

            scores_for_metric = outputs.detach() # 預設為 logits
            if task_type == 'multi-label, binary-class':
                scores_for_metric = torch.sigmoid(outputs).detach() # 多標籤用 sigmoid 機率
            
            all_labels_list.append(labels.cpu()) # 收集原始標籤 (CPU)
            all_scores_list.append(scores_for_metric.cpu()) # 收集模型輸出分數 (CPU)
            
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
    n_classes = 0

    if task == "multi-label, binary-class":
        n_classes = len(info['label'])
    elif task == "binary-class":
        n_classes = 2
    elif task == "multi-class" or task == "ordinal-regression":
        n_classes = len(info['label'])
    else:
        raise ValueError(f"未知的或不支援的任務類型: {task} for data_flag {args.data_flag}")

    print(f"資料集資訊: 通道數={n_channels}, 類別數={n_classes}, 任務類型={task}")

    model = SimpleCNN(in_channels=n_channels, num_classes=n_classes)
    
    try:
        model = load_model(model, model_full_path_dir, args.model_filename, device)
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print(f"請確保模型 '{os.path.join(model_full_path_dir, args.model_filename)}' 存在。")
        print("你可以先執行 train.py 來訓練並儲存一個模型。")
        return
    
    # model.to(device) # load_model 內部已經將模型移至 device

    test_y_true_np, test_y_score_np = test_model(model, test_loader, device, task)
    
    auc_value = 0.0
    acc_value = 0.0

    y_true_sklearn = test_y_true_np.squeeze()

    print_messages = [f"測試結果 ({args.data_flag}):"]

    if task == "binary-class":
        y_score_logits_tensor = torch.from_numpy(test_y_score_np)
        y_score_probs = torch.softmax(y_score_logits_tensor, dim=1)
        y_score_positive_class_probs_np = y_score_probs[:, 1].numpy()
        try:
            auc_value = roc_auc_score(y_true_sklearn, y_score_positive_class_probs_np)
        except ValueError as e:
            print(f"警告：計算 AUC 時出錯 ({e})。可能是所有樣本都屬於同一類別。將 AUC 設為 0.5。")
            auc_value = 0.5
        
        predicted_labels_np = np.argmax(test_y_score_np, axis=1)
        acc_value = accuracy_score(y_true_sklearn, predicted_labels_np)
        
        print_messages.append(f"  AUC: {auc_value:.4f}")
        print_messages.append(f"  Accuracy: {acc_value:.4f}")

    elif task == "multi-label, binary-class":
        try:
            auc_value = roc_auc_score(y_true_sklearn, test_y_score_np, average='macro')
        except ValueError as e:
            print(f"警告：計算多標籤 AUC 時出錯 ({e})。將 AUC 設為 0.5。")
            auc_value = 0.5
        
        predicted_labels_np = (test_y_score_np > 0.5).astype(int)
        acc_value = accuracy_score(y_true_sklearn, predicted_labels_np) # Subset accuracy
        
        print_messages.append(f"  AUC (Macro): {auc_value:.4f}")
        print_messages.append(f"  Accuracy (Subset): {acc_value:.4f}")

    else: # multi-class or ordinal-regression
        predicted_labels_np = np.argmax(test_y_score_np, axis=1)
        acc_value = accuracy_score(y_true_sklearn, predicted_labels_np)
        
        y_score_logits_tensor = torch.from_numpy(test_y_score_np)
        y_score_probs_np = torch.softmax(y_score_logits_tensor, dim=1).numpy()
        try:
            auc_value = roc_auc_score(y_true_sklearn, y_score_probs_np, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"警告：計算多類別 AUC 時出錯 ({e})。將 AUC 設為 0.0。")
            auc_value = 0.0
            
        print_messages.append(f"  Accuracy: {acc_value:.4f}")
        print_messages.append(f"  AUC (Macro OvR): {auc_value:.4f}")
    
    for msg in print_messages:
        print(msg)

    print("測試完成。")

if __name__ == '__main__':
    main()