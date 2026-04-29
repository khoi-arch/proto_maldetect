import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# 1. TÙY CHỈNH LOSS FUNCTION
# ==========================================
class FocalLoss(nn.Module):
    # Tăng smoothing lên 0.2 để giảm bớt sự cực đoan khi học 15 class khó
    def __init__(self, alpha=None, gamma=3.0, label_smoothing=0.2):
        super().__init__()
        self.alpha = alpha 
        self.gamma = gamma 
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
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
# 2. HÀM ĐÁNH GIÁ
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

            # Đổi cách gọi vì model giờ chỉ trả về embeddings (out)
            out = model(x)
            logits_binary = model.binary_head(out)
            logits_family = model.family_head(out)
            
            target_binary = (y > 0).long()
            target_family = y - 1
            malware_mask = (y > 0)
            
            loss_bin = nn.functional.cross_entropy(logits_binary, target_binary)
            
            if malware_mask.sum() > 0:
                loss_fam = criterion_family(logits_family[malware_mask], target_family[malware_mask])
                # Tính loss logic để theo dõi (không dùng để backward)
                loss = (loss_bin + loss_fam) / 2.0
            else:
                loss = loss_bin

            total_loss += loss.item()

            preds_binary = torch.argmax(logits_binary, dim=1)
            preds_family = torch.argmax(logits_family, dim=1) + 1
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
# 3. VÒNG LẶP HUẤN LUYỆN
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

    print("\n[*] Đang cấu hình CÔ LẬP GRADIENT (Gradient Isolation)...")
    
    malware_labels = train_labels[train_labels > 0] - 1
    classes = np.unique(malware_labels)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=malware_labels)
    alpha_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    
    # Label smoothing 0.2
    criterion_family = FocalLoss(alpha=alpha_weights, gamma=3.0, label_smoothing=0.2)

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

            # 1. Lấy vector đặc trưng tổng hợp từ Transformer
            out = model(x)

            target_binary = (y > 0).long() 
            target_family = y - 1          
            malware_mask = (y > 0)         

            # --- CƠ CHẾ CÔ LẬP GRADIENT ---
            
            # BƯỚC 1: Dạy cho Transformer biết cách phân biệt 15 mã độc
            loss_fam_tracker = 0
            if malware_mask.sum() > 0:
                logits_family = model.family_head(out)
                loss_fam = criterion_family(logits_family[malware_mask], target_family[malware_mask])
                # Ép toàn bộ mô hình cập nhật dựa trên độ khó của Family
                loss_fam.backward(retain_graph=True) 
                loss_fam_tracker = loss_fam.item()

            # BƯỚC 2: Cắt đứt kết nối, chỉ cho Binary Head học đoạn ngọn
            out_detached = out.detach() # CHỐT CHẶN: Gradient không đi xuống dưới chữ out này được nữa
            logits_binary = model.binary_head(out_detached)
            loss_bin = nn.functional.cross_entropy(logits_binary, target_binary)
            
            # Chỉ cập nhật trọng số của `self.binary_head`, Transformer giữ nguyên
            loss_bin.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step() 

            # Log loss tổng chỉ để xem cho vui, không dùng backward
            current_loss = loss_bin.item() + loss_fam_tracker
            total_loss += current_loss
            loop.set_postfix(loss=current_loss)

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