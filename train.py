import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import roc_auc_score, accuracy_score 

from data_utils import get_data_loader, SUPPORTED_2D_DATASETS
from model import SimpleCNN
from utils import save_model

def train_epoch(model, train_loader, optimizer, criterion, device, task_type):
    model.train()
    running_loss = 0.0
    all_labels_list = []
    all_scores_list = []

    for inputs, labels in tqdm(train_loader, desc="訓練中"):
        inputs_on_device, labels_on_device = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs_on_device) # logits

        current_loss_labels = labels_on_device
        scores_for_metric = outputs.detach() # 預設為 logits

        if task_type == 'multi-label, binary-class':
            current_loss_labels = current_loss_labels.to(torch.float32)
            loss = criterion(outputs, current_loss_labels)
            scores_for_metric = torch.sigmoid(outputs).detach() # 多標籤用 sigmoid 機率
        elif task_type == 'binary-class':
            current_loss_labels = current_loss_labels.squeeze().long()
            loss = criterion(outputs, current_loss_labels)
            # scores_for_metric 保持為 logits
        else: # multi-class or ordinal-regression
            current_loss_labels = current_loss_labels.squeeze().long()
            loss = criterion(outputs, current_loss_labels)
            # scores_for_metric 保持為 logits
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        all_labels_list.append(labels.cpu()) # 收集原始標籤 (CPU)
        all_scores_list.append(scores_for_metric.cpu()) # 收集模型輸出分數 (CPU)

    epoch_loss = running_loss / len(train_loader.dataset)
    
    final_y_true = torch.cat(all_labels_list, dim=0).numpy()
    final_y_score = torch.cat(all_scores_list, dim=0).numpy()
    
    return epoch_loss, final_y_true, final_y_score

def validate_epoch(model, val_loader, criterion, device, task_type):
    model.eval()
    running_loss = 0.0
    all_labels_list = []
    all_scores_list = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="驗證中"):
            inputs_on_device, labels_on_device = inputs.to(device), labels.to(device)
            outputs = model(inputs_on_device) # logits

            current_loss_labels = labels_on_device
            scores_for_metric = outputs.detach() # 預設為 logits

            if task_type == 'multi-label, binary-class':
                current_loss_labels = current_loss_labels.to(torch.float32)
                loss = criterion(outputs, current_loss_labels)
                scores_for_metric = torch.sigmoid(outputs).detach()
            elif task_type == 'binary-class':
                current_loss_labels = current_loss_labels.squeeze().long()
                loss = criterion(outputs, current_loss_labels)
                # scores_for_metric 保持為 logits
            else: # multi-class or ordinal-regression
                current_loss_labels = current_loss_labels.squeeze().long()
                loss = criterion(outputs, current_loss_labels)
                # scores_for_metric 保持為 logits
            
            running_loss += loss.item() * inputs.size(0)

            all_labels_list.append(labels.cpu())
            all_scores_list.append(scores_for_metric.cpu())
            
    epoch_loss = running_loss / len(val_loader.dataset)
    final_y_true = torch.cat(all_labels_list, dim=0).numpy()
    final_y_score = torch.cat(all_scores_list, dim=0).numpy()

    return epoch_loss, final_y_true, final_y_score


def main():
    parser = argparse.ArgumentParser(description="MedMNIST 2D 資料集訓練腳本")
    parser.add_argument('--data_flag', type=str, default='pneumoniamnist',
                        help=f"MedMNIST 資料集標籤 (僅限 28x28)。可選: {', '.join(SUPPORTED_2D_DATASETS)}")
    parser.add_argument('--output_root', type=str, default='./trained_models',
                        help="儲存訓練模型權重的目錄")
    parser.add_argument('--num_epochs', type=int, default=10, help="訓練的週期數")
    parser.add_argument('--lr', type=float, default=0.001, help="學習率")
    parser.add_argument('--batch_size', type=int, default=128, help="批次大小")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader 的工作執行緒數量")
    parser.add_argument('--data_aug', action='store_true', help="是否使用資料增強")
    parser.add_argument('--model_filename', type=str, default="model.pth", help="儲存的模型檔案名稱")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    print(f"載入資料集: {args.data_flag}...")
    train_loader, info = get_data_loader(args.data_flag, args.batch_size, 'train', 
                                         num_workers=args.num_workers, data_aug=args.data_aug)
    val_loader, _ = get_data_loader(args.data_flag, args.batch_size, 'val', 
                                    num_workers=args.num_workers, data_aug=False)
    
    n_channels = info['n_channels']
    task = info['task']
    
    n_classes = 0
    criterion = None
    primary_metric_for_selection = "" # 用於模型選擇和打印的主要指標名稱

    if task == "multi-label, binary-class":
        n_classes = len(info['label'])
        criterion = nn.BCEWithLogitsLoss()
        primary_metric_for_selection = "AUC (Macro)"
    elif task == "binary-class":
        n_classes = 2 
        criterion = nn.CrossEntropyLoss()
        primary_metric_for_selection = "AUC"
    elif task == "multi-class" or task == "ordinal-regression": 
        n_classes = len(info['label'])
        criterion = nn.CrossEntropyLoss()
        primary_metric_for_selection = "Accuracy"
    else:
        raise ValueError(f"未知的或不支援的任務類型: {task} for data_flag {args.data_flag}")

    print(f"資料集資訊: 通道數={n_channels}, 類別數={n_classes}, 任務類型={task}")

    model = SimpleCNN(in_channels=n_channels, num_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_metric = 0.0 # 初始化最佳指標值

    print(f"開始訓練 {args.data_flag}...")
    for epoch in range(args.num_epochs):
        # 訓練時返回的 y_true 和 y_score 我們在這裡不直接用，但函數內部會計算損失
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, criterion, device, task)
        val_loss, val_y_true_np, val_y_score_np = validate_epoch(model, val_loader, criterion, device, task)
        
        auc_value = 0.0
        acc_value = 0.0
        current_epoch_primary_metric_value = 0.0 # 這個 epoch 的主要評估指標值

        y_true_sklearn = val_y_true_np.squeeze() # 確保 y_true 是 1D

        print_msg_parts = [
            f"Epoch [{epoch+1}/{args.num_epochs}]",
            f"訓練損失: {train_loss:.4f}",
            f"驗證損失: {val_loss:.4f}"
        ]

        if task == "binary-class":
            y_score_logits_tensor = torch.from_numpy(val_y_score_np)
            y_score_probs = torch.softmax(y_score_logits_tensor, dim=1)
            y_score_positive_class_probs_np = y_score_probs[:, 1].numpy()
            try:
                auc_value = roc_auc_score(y_true_sklearn, y_score_positive_class_probs_np)
            except ValueError as e:
                print(f"警告：Epoch {epoch+1} 計算 AUC 時出錯 ({e})。將 AUC 設為 0.5。")
                auc_value = 0.5
            predicted_labels_np = np.argmax(val_y_score_np, axis=1)
            acc_value = accuracy_score(y_true_sklearn, predicted_labels_np)
            
            current_epoch_primary_metric_value = auc_value
            print_msg_parts.extend([f"驗證 AUC: {auc_value:.4f}", f"驗證 Accuracy: {acc_value:.4f}"])

        elif task == "multi-label, binary-class":
            # val_y_score_np 已經是 sigmoid 機率
            try:
                auc_value = roc_auc_score(y_true_sklearn, val_y_score_np, average='macro')
            except ValueError as e:
                print(f"警告：Epoch {epoch+1} 計算多標籤 AUC 時出錯 ({e})。將 AUC 設為 0.5。")
                auc_value = 0.5
            predicted_labels_np = (val_y_score_np > 0.5).astype(int)
            acc_value = accuracy_score(y_true_sklearn, predicted_labels_np) # Subset accuracy
            
            current_epoch_primary_metric_value = auc_value
            print_msg_parts.extend([f"驗證 AUC (Macro): {auc_value:.4f}", f"驗證 Accuracy (Subset): {acc_value:.4f}"])

        else: # multi-class or ordinal-regression
            predicted_labels_np = np.argmax(val_y_score_np, axis=1)
            acc_value = accuracy_score(y_true_sklearn, predicted_labels_np)
            
            y_score_logits_tensor = torch.from_numpy(val_y_score_np)
            y_score_probs_np = torch.softmax(y_score_logits_tensor, dim=1).numpy()
            try:
                auc_value = roc_auc_score(y_true_sklearn, y_score_probs_np, multi_class='ovr', average='macro')
            except ValueError as e:
                print(f"警告：Epoch {epoch+1} 計算多類別 AUC 時出錯 ({e})。將 AUC 設為 0.0。")
                auc_value = 0.0
            
            current_epoch_primary_metric_value = acc_value
            print_msg_parts.extend([f"驗證 Accuracy: {acc_value:.4f}", f"驗證 AUC (Macro OvR): {auc_value:.4f}"])
        
        print(", ".join(print_msg_parts))

        if current_epoch_primary_metric_value > best_val_metric :
            best_val_metric = current_epoch_primary_metric_value
            model_save_path = os.path.join(args.output_root, args.data_flag)
            save_model(model, model_save_path, args.model_filename)
            print(f"在 Epoch {epoch+1} 找到新的最佳模型，驗證 {primary_metric_for_selection}: {best_val_metric:.4f}")

    print("訓練完成。")
    print(f"最佳驗證 {primary_metric_for_selection}: {best_val_metric:.4f}")
    print(f"模型已儲存於 {os.path.join(args.output_root, args.data_flag, args.model_filename)}")

if __name__ == '__main__':
    main()