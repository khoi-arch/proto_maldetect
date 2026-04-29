import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# 1. TÙY CHỈNH LOSS FUNCTION (FOCAL LOSS CÓ LABEL SMOOTHING)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha 
        self.gamma = gamma 
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Tính CE có Label Smoothing để chống overconfident
        ce_loss = nn.functional.cross_entropy(
            logits, targets, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = focal_loss * alpha_t
            
        return focal_loss.mean()


# ==========================================
# 2. HÀM ĐÁNH GIÁ (HIERARCHICAL EVALUATE)
# ==========================================
def evaluate(model, loader, device, criterion_family, label_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits_binary, logits_family = model(x)
            
            # --- TÍNH LOSS Y HỆT LÚC TRAIN ---
            target_binary = (y > 0).long()
            target_family = y - 1
            malware_mask = (y > 0)
            
            loss_bin = nn.functional.cross_entropy(logits_binary, target_binary)
            
            if malware_mask.sum() > 0:
                loss_fam = criterion_family(logits_family[malware_mask], target_family[malware_mask])
                loss = 0.3 * loss_bin + 0.7 * loss_fam
            else:
                loss = loss_bin

            total_loss += loss.item()

            # --- LOGIC GỘP DỰ ĐOÁN (INFERENCE) ---
            # 1. Đoán nhị phân trước
            preds_binary = torch.argmax(logits_binary, dim=1)
            # 2. Đoán family (cộng 1 để bù lại index của Benign)
            preds_family = torch.argmax(logits_family, dim=1) + 1
            
            # 3. Chốt kết quả: Nếu đoán Binary = 0 thì là 0. Ngược lại lấy kết quả Family.
            final_preds = torch.where(preds_binary == 0, torch.zeros_like(preds_binary), preds_family)

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
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
    train_labels,  
    label_names,   
    device,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5
):
    model = model.to(device)

    print("\n[*] Đang cấu hình Masked Focal Loss cho nhánh Family...")
    
    # 1. CHỈ tính weights cho 15 loại Malware (Bỏ Benign ra khỏi trọng số)
    malware_labels = train_labels[train_labels > 0] - 1
    classes = np.unique(malware_labels)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=malware_labels)
    alpha_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    
    # Khởi tạo Focal Loss với gamma=3.0 và smoothing=0.1 chuyên trị mã độc khó
    criterion_family = FocalLoss(alpha=alpha_weights, gamma=3.0, label_smoothing=0.1)

    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True, 
        num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1 
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

            # Forward ra 2 chóp
            logits_binary, logits_family = model(x)

            # Tạo nhãn phân tầng
            target_binary = (y > 0).long() # 0 = Benign, 1 = Malware
            target_family = y - 1          # Dịch [1..15] thành [0..14]
            malware_mask = (y > 0)         # Mặt nạ lọc malware

            # 1. Tính Loss cho Binary Head (Dễ học, dùng CE thường)
            loss_bin = nn.functional.cross_entropy(logits_binary, target_binary)
            
            # 2. Tính Loss cho Family Head (Khó học, dùng Focal + Mask)
            if malware_mask.sum() > 0:
                loss_fam = criterion_family(logits_family[malware_mask], target_family[malware_mask])
                # Trọng số ưu tiên nhánh Family để ép mô hình học đặc trưng mã độc
                loss = 0.3 * loss_bin + 0.7 * loss_fam
            else:
                loss = loss_bin # Trường hợp batch xui xẻo toàn Benign

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step() 

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_loss, val_acc, val_f1, val_report = evaluate(model, val_loader, device, criterion_family, label_names)

        print(f"\n📊 Epoch {epoch+1}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1 Macro: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0

            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model (F1: {best_f1:.4f})")
            print("\n🔍 CLASSIFICATION REPORT (BEST EPOCH):")
            print(val_report)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n⛔ Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\n🏆 Best F1 Macro: {best_f1:.4f}")