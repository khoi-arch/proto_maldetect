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
    # Dùng smoothing 0.2 để giảm bớt sự cực đoan khi học 15 class khó
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

            logits_binary, logits_family = model(x)
            
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
    epochs=50,       # Để 50 epoch do dùng Plateau cần thời gian
    lr=2e-4,         # GIẢM LR KHỞI ĐIỂM
    weight_decay=1e-3, # TĂNG REGULARIZATION
    patience=8       # Tăng patience để đợi LR scheduler hoạt động
):
    model = model.to(device)

    print("\n[*] Đang cấu hình Masked Focal Loss & ReduceLROnPlateau...")
    
    malware_labels = train_labels[train_labels > 0] - 1
    classes = np.unique(malware_labels)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=malware_labels)
    alpha_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    
    criterion_family = FocalLoss(alpha=alpha_weights, gamma=3.0, label_smoothing=0.2)

    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True, 
        num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # DÙNG REDUCELRONPLATEAU THAY VÌ ONECYCLELR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # Maximize F1
        factor=0.5,      # Giảm LR đi một nửa
        patience=3,      # Đợi 3 epoch không tiến bộ thì giảm LR
        verbose=True
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

            logits_binary, logits_family = model(x)

            target_binary = (y > 0).long() 
            target_family = y - 1          
            malware_mask = (y > 0)         

            loss_bin = nn.functional.cross_entropy(logits_binary, target_binary)
            
            if malware_mask.sum() > 0:
                loss_fam = criterion_family(logits_family[malware_mask], target_family[malware_mask])
                loss = 0.3 * loss_bin + 0.7 * loss_fam
            else:
                loss = loss_bin

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            # XÓA BỎ scheduler.step() Ở ĐÂY RỒI NHÉ

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_loss, val_acc, val_f1, val_report = evaluate(model, val_loader, device, criterion_family, label_names)

        print(f"\n📊 Epoch {epoch+1}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1 Macro: {val_f1:.4f}")

        # GỌI SCHEDULER Ở CUỐI EPOCH (Dựa vào val_f1)
        scheduler.step(val_f1)

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