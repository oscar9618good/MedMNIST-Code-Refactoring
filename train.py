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
    # y_true 和 y_score 在迴圈內部處理，確保維度和類型正確
    all_labels_list = []
    all_scores_list = []

    for inputs, labels in tqdm(train_loader, desc="訓練中"):
        inputs, labels_on_device = inputs.to(device), labels.to(device) # labels_on_device 用於損失計算

        optimizer.zero_grad()
        outputs = model(inputs) # logits

        current_loss_labels = labels_on_device # 用於損失計算的標籤
        if task_type == 'multi-label, binary-class':
            current_loss_labels = current_loss_labels.to(torch.float32)
            loss = criterion(outputs, current_loss_labels)
            # 對於多標籤，scores 通常是 sigmoid 後的機率
            scores_for_metric = torch.sigmoid(outputs).detach()
        elif task_type == 'binary-class':
            current_loss_labels = current_loss_labels.squeeze().long()
            loss = criterion(outputs, current_loss_labels)
            # 對於二元分類，scores 是 logits
            scores_for_metric = outputs.detach()
        else: # multi-class
            current_loss_labels = current_loss_labels.squeeze().long()
            loss = criterion(outputs, current_loss_labels)
            # 對於多類別，scores 是 logits
            scores_for_metric = outputs.detach()
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        # 收集原始標籤 (CPU) 和模型輸出分數 (CPU) 以便後續處理
        all_labels_list.append(labels.cpu()) # labels 是原始從 dataloader 來的
        all_scores_list.append(scores_for_metric.cpu())

    epoch_loss = running_loss / len(train_loader.dataset)
    
    # 在函數末尾將列表中的 tensors 合併並轉換為 numpy arrays
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
            inputs, labels_on_device = inputs.to(device), labels.to(device)
            outputs = model(inputs) # logits

            current_loss_labels = labels_on_device
            if task_type == 'multi-label, binary-class':
                current_loss_labels = current_loss_labels.to(torch.float32)
                loss = criterion(outputs, current_loss_labels)
                scores_for_metric = torch.sigmoid(outputs).detach()
            elif task_type == 'binary-class':
                current_loss_labels = current_loss_labels.squeeze().long()
                loss = criterion(outputs, current_loss_labels)
                scores_for_metric = outputs.detach()
            else: # multi-class
                current_loss_labels = current_loss_labels.squeeze().long()
                loss = criterion(outputs, current_loss_labels)
                scores_for_metric = outputs.detach()
            
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
    
    if task == "multi-label, binary-class":
        n_classes = len(info['label'])
    elif 'multi-class' in task:
        n_classes = len(info['label'])
    elif 'binary-class' in task:
        n_classes = 2
    else:
        raise ValueError(f"未知的任務類型: {task}")

    print(f"資料集資訊: 通道數={n_channels}, 類別數={n_classes}, 任務類型={task}")

    model = SimpleCNN(in_channels=n_channels, num_classes=n_classes).to(device)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_metric = 0.0
    primary_metric_for_selection = "" # 用於模型選擇的指標名稱

    if task == "binary-class":
        primary_metric_for_selection = "AUC"
    elif task == "multi-label, binary-class":
        primary_metric_for_selection = "AUC (Macro)"
    else: # multi-class
        primary_metric_for_selection = "Accuracy"

    print(f"開始訓練 {args.data_flag}...")
    for epoch in range(args.num_epochs):
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, criterion, device, task)
        val_loss, val_y_true_np, val_y_score_np = validate_epoch(model, val_loader, criterion, device, task)
        
        auc_value = 0.0
        acc_value = 0.0
        current_epoch_primary_metric = 0.0 # 這個 epoch 的主要評估指標值

        y_true_sklearn = val_y_true_np.squeeze()

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
            current_epoch_primary_metric = auc_value # 主要指標是 AUC

            print_msg = (f"Epoch [{epoch+1}/{args.num_epochs}], "
                         f"訓練損失: {train_loss:.4f}, 驗證損失: {val_loss:.4f}, "
                         f"驗證 AUC: {auc_value:.4f}, 驗證 Accuracy: {acc_value:.4f}")

        elif task == "multi-label, binary-class":
            try:
                auc_value = roc_auc_score(y_true_sklearn, val_y_score_np, average='macro')
            except ValueError as e:
                print(f"警告：Epoch {epoch+1} 計算多標籤 AUC 時出錯 ({e})。將 AUC 設為 0.5。")
                auc_value = 0.5
            predicted_labels_np = (val_y_score_np > 0.5).astype(int)
            acc_value = accuracy_score(y_true_sklearn, predicted_labels_np)
            current_epoch_primary_metric = auc_value # 主要指標是 AUC (Macro)

            print_msg = (f"Epoch [{epoch+1}/{args.num_epochs}], "
                         f"訓練損失: {train_loss:.4f}, 驗證損失: {val_loss:.4f}, "
                         f"驗證 AUC (Macro): {auc_value:.4f}, 驗證 Accuracy (Subset): {acc_value:.4f}")
        else: # multi-class
            predicted_labels_np = np.argmax(val_y_score_np, axis=1)
            acc_value = accuracy_score(y_true_sklearn, predicted_labels_np)
            
            y_score_logits_tensor = torch.from_numpy(val_y_score_np)
            y_score_probs_np = torch.softmax(y_score_logits_tensor, dim=1).numpy()
            try:
                auc_value = roc_auc_score(y_true_sklearn, y_score_probs_np, multi_class='ovr', average='macro')
            except ValueError as e:
                print(f"警告：Epoch {epoch+1} 計算多類別 AUC 時出錯 ({e})。將 AUC 設為 0.0。")
                auc_value = 0.0
            current_epoch_primary_metric = acc_value # 主要指標是 Accuracy

            print_msg = (f"Epoch [{epoch+1}/{args.num_epochs}], "
                         f"訓練損失: {train_loss:.4f}, 驗證損失: {val_loss:.4f}, "
                         f"驗證 Accuracy: {acc_value:.4f}, 驗證 AUC (Macro OvR): {auc_value:.4f}")
        
        print(print_msg) # <--- 修改後的打印語句

        if current_epoch_primary_metric > best_val_metric :
            best_val_metric = current_epoch_primary_metric
            model_save_path = os.path.join(args.output_root, args.data_flag)
            save_model(model, model_save_path, args.model_filename)
            print(f"在 Epoch {epoch+1} 找到新的最佳模型，驗證 {primary_metric_for_selection}: {best_val_metric:.4f}")

    print("訓練完成。")
    print(f"最佳驗證 {primary_metric_for_selection}: {best_val_metric:.4f}")
    print(f"模型已儲存於 {os.path.join(args.output_root, args.data_flag, args.model_filename)}")

if __name__ == '__main__':
    main()