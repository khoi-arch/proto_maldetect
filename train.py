import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# 1. HÀM ĐÁNH GIÁ (KÈM POST-HOC LOGIT ADJUSTMENT)
# ==========================================
def evaluate(model, loader, device, criterion, class_prior=None, label_names=None):
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

            # --- SOTA TRICK: LOGIT ADJUSTMENT TẠI BƯỚC INFERENCE ---
            if class_prior is not None:
                # Cộng log_prior để bù lại độ vặn xoắn do class_weights sinh ra lúc train
                adjusted_logits = logits + torch.log(class_prior + 1e-9)
                preds = torch.argmax(adjusted_logits, dim=1)
            else:
                preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    if label_names is not None:
        report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
    else:
        report = classification_report(all_labels, all_preds, zero_division=0)

    # Đếm phân phối số lượng predict để check xem model có bị collapse không
    num_classes = 16 if label_names is None else len(label_names)
    pred_dist = np.bincount(all_preds, minlength=num_classes)

    return total_loss / len(loader), acc, f1, report, pred_dist


# ==========================================
# 2. VÒNG LẶP HUẤN LUYỆN
# ==========================================
def train_model(
    model,
    train_loader,
    val_loader,
    train_labels,  
    label_names,   
    device,
    epochs=50,       
    lr=2e-4,         
    weight_decay=1e-3, 
    patience=8       
):
    model = model.to(device)

    print("\n[*] Đang cấu hình Weighted CrossEntropy & Logit Adjustment...")
    
    # --- 1. TÍNH TOÁN CLASS PRIOR VÀ TRỌNG SỐ ---
    class_counts = np.bincount(train_labels)
    classes = np.unique(train_labels)
    
    # Prior thực tế của tập Train (dùng để Adjustment lúc Evaluate)
    class_prior_np = class_counts / class_counts.sum()
    class_prior = torch.tensor(class_prior_np, dtype=torch.float32).to(device)
    
    # Dùng class_weight chuẩn (để ép mô hình chú ý class hiếm lúc backward)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    alpha_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    # --- 2. BỎ SAMPLER, TRẢ VỀ SHUFFLE=TRUE ---
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True, 
        num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0
    )

    # --- 3. DÙNG CROSS ENTROPY THUẦN TÚY KÈM WEIGHTS (Vứt bỏ Focal Loss) ---
    criterion = nn.CrossEntropyLoss(weight=alpha_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      
        factor=0.5,      
        patience=3       
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Truyền thêm class_prior vào Evaluate để thực hiện Logit Adjustment
        val_loss, val_acc, val_f1, val_report, pred_dist = evaluate(
            model, val_loader, device, criterion, class_prior, label_names
        )

        print(f"\n📊 Epoch {epoch+1}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1 Macro: {val_f1:.4f}")
        
        # IN PHÂN PHỐI PREDICT ĐỂ PHÁT HIỆN MODEL COLLAPSE
        print(f"\n🔍 Prediction Distribution: \n{pred_dist}")

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