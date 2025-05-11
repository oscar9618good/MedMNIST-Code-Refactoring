import torch
import os

def save_model(model, path, filename="model.pth"):
    """
    儲存 PyTorch 模型。

    Args:
        model (torch.nn.Module): 要儲存的模型。
        path (str): 模型儲存的路徑目錄。
        filename (str): 模型檔案的名稱。
    """
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, filename)
    torch.save(model.state_dict(), model_path)
    print(f"模型已儲存至 {model_path}")

def load_model(model, path, filename="model.pth", device='cpu'):
    """
    載入 PyTorch 模型權重。

    Args:
        model (torch.nn.Module): 模型架構 (權重將載入其中)。
        path (str): 模型權重檔案的路徑目錄。
        filename (str): 模型檔案的名稱。
        device (str): 要載入模型的設備 ('cpu' 或 'cuda')。

    Returns:
        torch.nn.Module: 載入權重後的模型。
    """
    model_path = os.path.join(path, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"模型已從 {model_path} 載入並移至 {device}")
    return model

if __name__ == '__main__':
    # 測試 utils.py
    print("測試 utils.py...")
    from model import SimpleCNN # 假設 model.py 在同一個目錄或 PYTHONPATH

    # 建立一個假模型
    test_model = SimpleCNN(in_channels=1, num_classes=2)
    test_save_path = "./test_model_weights"
    
    # 測試儲存
    save_model(test_model, test_save_path, "test_cnn.pth")

    # 建立一個新的模型實例來載入權重
    loaded_model = SimpleCNN(in_channels=1, num_classes=2)
    try:
        load_model(loaded_model, test_save_path, "test_cnn.pth")
        print("模型載入測試成功。")
    except Exception as e:
        print(f"模型載入測試失敗: {e}")
    
    # 清理測試檔案
    if os.path.exists(os.path.join(test_save_path, "test_cnn.pth")):
        os.remove(os.path.join(test_save_path, "test_cnn.pth"))
    if os.path.exists(test_save_path):
        os.rmdir(test_save_path)
    print("測試檔案已清理。")