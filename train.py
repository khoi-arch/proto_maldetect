import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# 1. TÙY CHỈNH LOSS FUNCTION (FOCAL LOSS)
# ==========================================
class FocalLoss(nn.Module):
    """
    Focal Loss: Ép mô hình tập trung vào các mẫu khó (mã độc hiếm) 
    và phớt lờ các mẫu dễ (Benign chiếm đa số).
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha # Class weights tensor
        self.gamma = gamma # Tham số phạt (thường dùng 2.0)

    def forward(self, logits, targets):
        # 1. Tính CE cơ bản KHÔNG có weight để trích xuất đúng pt
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        
        # Tính xác suất (pt) mà mô hình dự đoán cho class đúng
        pt = torch.exp(-ce_loss)
        
        # 2. Tính Focal Loss tiêu chuẩn
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 3. Áp dụng Alpha weights thủ công vào từng mẫu
        if self.alpha is not None:
            # Lấy weight tương ứng với nhãn đúng của từng sample trong batch
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = focal_loss * alpha_t
            
        return focal_loss.mean()


# ==========================================
# 2. HÀM ĐÁNH GIÁ (EVALUATE)
# ==========================================
def evaluate(model, loader, device, criterion, label_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Sinh báo cáo chi tiết cho từng class
    if label_names is not None:
        report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
    else:
        report = classification_report(all_labels, all_preds, zero_division=0)

    return total_loss / len(loader), acc, f1, report


# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN (TRAIN LOOP)
# ==========================================
def train_model(
    model,
    train_loader,
    val_loader,
    train_labels,  # mảng numpy chứa nhãn tập train
    label_names,   # list tên các class để in report
    device,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5
):
    model = model.to(device)

    print("\n[*] Đang cấu hình Class Weights & Focal Loss...")
    
    # 1. Tính toán Trọng số cho Focal Loss
    classes = np.unique(train_labels)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    alpha_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    
    # Dùng Focal Loss đã được sửa lỗi
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)

    # 2. Bỏ Sampler, dùng DataLoader mặc định với shuffle=True để tránh xung đột kép
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True, # Bật shuffle để xáo trộn dữ liệu
        num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler: Dùng OneCycleLR thay cho CosineAnnealingLR để có cơ chế Warm-up
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1 # Dùng 10% thời gian đầu để Warm-up
    )

    best_f1 = 0
    patience_counter = 0

    os.makedirs("log", exist_ok=True)
    save_path = os.path.join("log", "best_model.pt")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()

            # Gradient clipping bảo vệ Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            # Chú ý: Cập nhật scheduler OneCycleLR sau mỗi batch (thay vì sau mỗi epoch)
            scheduler.step() 

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Đánh giá trên tập Validation
        val_loss, val_acc, val_f1, val_report = evaluate(model, val_loader, device, criterion, label_names)

        print(f"\n📊 Epoch {epoch+1}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1 Macro: {val_f1:.4f}")

        # Early Stopping dựa trên F1 Macro
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0

            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model (F1: {best_f1:.4f})")
            
            # In chi tiết Classification Report để theo dõi class hiếm
            print("\n🔍 CLASSIFICATION REPORT (BEST EPOCH):")
            print(val_report)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n⛔ Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\n🏆 Best F1 Macro: {best_f1:.4f}")