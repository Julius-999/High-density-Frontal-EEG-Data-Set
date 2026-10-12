import os
import random
import math
import numpy as np
from collections import Counter
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================
# 0. Hyperparameters
# =====================
SEED = 2026
batch_size = 64
lr = 2e-4
max_epochs = 20
d_model = 128
nhead = 8
num_layers = 8
ff_dim = 256
dropout = 0.15
save_path = "best_model/final_eeg_transformer.pt"
T_0 = 5
T_mult = 2
eta_min = 1e-6
CHANNEL_NUMBER = 48 * 5

# =====================
# 1. Transformer model
# =====================
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        """
        eeg_data: numpy [N, T, C]
        labels:   numpy [N]
        """
        self.data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_sample = torch.tensor(self.data[idx], dtype=torch.float32)  # [T, C]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return eeg_sample, label

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class EEGTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        in_channels,
        num_classes=3,
        cls_token=True
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pos = PositionalEncoding(d_model)
        self.cls_token = cls_token
        if cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.proj(x)
        if self.cls_token:
            B = x.size(0)
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        x = self.encoder(x)
        feat = x[:, 0, :] if self.cls_token else x.mean(dim=1)
        feat = self.norm(feat)
        logits = self.head(feat)
        return logits

# =====================
# 2. Training / evaluation utilities
# =====================
def compute_mean_std(loader):
    feats = []
    for x, _ in loader:
        feats.append(x.reshape(-1, x.shape[-1]))
    feats = torch.cat(feats, dim=0)
    return feats.mean(0), feats.std(0)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = (x - mean) / (std + 1e-8)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    acc = correct / max(total, 1)

    from sklearn.metrics import f1_score
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    macro_f1 = f1_score(y_true, y_pred, average='macro') if y_true.size else 0.0
    return acc, macro_f1, y_true, y_pred

def class_weights_from_labels(labels, num_classes):
    cnt = Counter(labels.tolist() if isinstance(labels, np.ndarray) else labels)
    total = sum(cnt.values())
    weights = []
    for c in range(num_classes):
        freq = cnt.get(c, 0) / total if total > 0 else 0
        w = 0.0 if freq == 0 else 1.0 / freq
        weights.append(w)
    s = sum(weights)
    if s > 0:
        weights = [w * len(weights) / s for w in weights]
    return torch.tensor(weights, dtype=torch.float32)

# =====================
# 3. Data loading
# =====================
def create_train_dataset_label(base_path,file_list,label):
    data_list = []
    for file in file_list:
        data_list.append(np.load(os.path.join(base_path,file)))
    X = np.concatenate(data_list, axis=0)
    y = np.full(shape=(X.shape[0],), fill_value=label, dtype=np.int64)
    return X, y

def create_test_dataset_label(base_path,file_list):
    negative_file = os.path.join(base_path,file_list[0])
    neutral_file = os.path.join(base_path, file_list[1])
    positive_file = os.path.join(base_path, file_list[2])
    neg_X = np.load(negative_file)
    neg_y = np.full(shape=(neg_X.shape[0],), fill_value=0, dtype=np.int64)
    neu_X = np.load(neutral_file)
    neu_y = np.full(shape=(neu_X.shape[0],), fill_value=1, dtype=np.int64)
    pos_X = np.load(positive_file)
    pos_y = np.full(shape=(pos_X.shape[0],), fill_value=2, dtype=np.int64)
    test_X = np.concatenate([neg_X, neu_X, pos_X],axis=0)
    test_y = np.concatenate([neg_y, neu_y, pos_y],axis=0)
    perm = np.random.permutation(len(test_X))
    test_X = test_X[perm]
    test_y = test_y[perm]
    return test_X, test_y

negative_file_list = ["negative01.npy", "negative02.npy", "negative03.npy", "negative04.npy"]
neutral_file_list  = ["neutral01.npy",  "neutral02.npy",  "neutral03.npy",  "neutral04.npy"]
positive_file_list = ["positive01.npy", "positive02.npy", "positive03.npy", "positive04.npy"]
all_results = []
subjects = [
    "sub1", "sub2", "sub3", "sub4", "sub5",
    "sub6", "sub7", "sub8", "sub9", "sub10",
    "sub11", "sub12", "sub13", "sub14", "sub15"
]

all_results = []
BP = r"HF-EEG/STFT-features"
for j in range(len(subjects)):
    BASE_PATH = os.path.join(BP,subjects[j])
    for neg_test in negative_file_list:
        for neu_test in neutral_file_list:
            for pos_test in positive_file_list:
                test_file_list = [neg_test, neu_test, pos_test]
                train_negative_file_list = [f for f in negative_file_list if f != neg_test]
                train_neutral_file_list  = [f for f in neutral_file_list  if f != neu_test]
                train_positive_file_list = [f for f in positive_file_list if f != pos_test]
                print("===============================================================")
                print(f"subject: {subjects[j]} | Test File: {test_file_list[0]}, {test_file_list[1]}, {test_file_list[2]}")

                train_X_neg,train_y_neg = create_train_dataset_label(BASE_PATH,train_negative_file_list,0)
                train_X_neu,train_y_neu = create_train_dataset_label(BASE_PATH,train_neutral_file_list,1)
                train_X_pos,train_y_pos = create_train_dataset_label(BASE_PATH,train_positive_file_list,2)
                train_X = np.concatenate([train_X_neg, train_X_neu, train_X_pos],axis=0)
                train_y = np.concatenate([train_y_neg, train_y_neu, train_y_pos],axis=0)
                perm = np.random.permutation(len(train_X))
                train_X = train_X[perm]
                train_y = train_y[perm]
                test_X,test_y = create_test_dataset_label(BASE_PATH,test_file_list)
                train_dataset = EEGDataset(train_X, train_y)
                test_dataset = EEGDataset(test_X, test_y)

                set_seed(SEED)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("Device:", device)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
                mean, std = compute_mean_std(train_loader)
                mean, std = mean.to(device), std.to(device)

                class_w = class_weights_from_labels(train_dataset.labels, 3).to(device)
                print("Class weights:", class_w.detach().cpu().numpy().round(3))

                model = EEGTransformer(
                    in_channels=CHANNEL_NUMBER,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=T_0,
                    T_mult=T_mult,
                    eta_min=eta_min
                )

                criterion = nn.CrossEntropyLoss(weight=class_w)

                scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

                for epoch in range(max_epochs):
                    model.train()
                    total_loss = 0.0
                    num_batches = len(train_loader)
                    for batch_idx, (x, y) in enumerate(train_loader):
                        x, y = x.to(device), y.to(device)
                        x = (x - mean) / (std + 1e-8)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            logits = model(x)
                            loss = criterion(logits, y)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step(epoch + batch_idx / num_batches)
                        total_loss += loss.item() * y.size(0)
                    train_loss = total_loss / max(len(train_dataset), 1)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1:02d} | TrainLoss {train_loss:.4f} | LR {current_lr:.2e}")

                torch.save({
                    'model': model.state_dict(),
                    'cfg': {'d_model': d_model, 'nhead': nhead, 'num_layers': num_layers,
                            'ff_dim': ff_dim, 'dropout': dropout}
                }, save_path)

                if os.path.exists(save_path):
                    ckpt = torch.load(save_path, map_location=device)
                    model.load_state_dict(ckpt['model'])

                acc, f1, y_true, y_pred = evaluate(model, test_loader, device)
                print(f"[Final Model Eval] Acc={acc:.4f} | MacroF1={f1:.4f}")
                all_results.append({
                    "subject": subjects[j],
                    "test_negative": test_file_list[0],
                    "test_neutral": test_file_list[1],
                    "test_positive": test_file_list[2],
                    "accuracy": float(acc),
                    "macro_f1": float(f1)
                })
df_results = pd.DataFrame(all_results)
excel_path = "transformer_results.xlsx"
df_results.to_excel(excel_path, index=False)
print(f"\n✅ All experiment results saved to: {excel_path}")
