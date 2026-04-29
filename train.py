import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


# ==========================================
# UTILS
# ==========================================
def compute_class_weights(labels):
    """Tính class weight để xử lý imbalance"""
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)

# 👉 FIX 1: Nhận thêm biến 'criterion' từ ngoài vào
def evaluate(model, loader, device, criterion):
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

    return total_loss / len(loader), acc, f1


# ==========================================
# TRAIN LOOP
# ==========================================
def train_model(
    model,
    train_loader,
    val_loader,
    train_labels,  # truyền từ dataset
    device,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5
):
    model = model.to(device)

    # 🔥 Class imbalance handling
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0
    patience_counter = 0

    # 👉 FIX 2: Tạo đường dẫn lưu model vào thư mục log/
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

            # 🔥 Gradient clipping (rất quan trọng cho transformer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        # 👉 FIX 1 (Tiếp): Truyền criterion vào để Val Loss đồng bộ với Train Loss
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)

        print(f"\n📊 Epoch {epoch+1}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # ===== EARLY STOPPING =====
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0

            # 👉 FIX 2 (Tiếp): Lưu vào thư mục log/
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

    print(f"\n🏆 Best F1: {best_f1:.4f}")