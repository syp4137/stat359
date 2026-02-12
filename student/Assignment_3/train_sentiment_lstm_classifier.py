import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import datasets

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

print("\n========== Loading Dataset ==========")
dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
data = pd.DataFrame(dataset["train"])   # columns: sentence, label
print("Example row:", data.iloc[0].to_dict())
print("Label counts:\n", data["label"].value_counts().sort_index())

sentences = data["sentence"].astype(str).tolist()
y = data["label"].to_numpy()

print("\n========== Stratified Split ==========")
X_idx = np.arange(len(y))
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_idx, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

print("\n========== Class Weights (for imbalance) ==========")
# inverse frequency weights: w_c = N / (K * n_c)
K = len(np.unique(y))
counts = np.bincount(y_train, minlength=K)
N = counts.sum()
class_weights = (N / (K * counts)).astype(np.float32)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
print("counts:", counts, "weights:", class_weights)


import gensim.downloader as api
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n========== Loading FastText (Gensim) ==========")
ft = api.load("fasttext-wiki-news-subwords-300")
EMB_DIM = 300
MAX_LEN = 32
print("FastText loaded.")


def sentence_to_seq(sentence: str, ft_model, max_len=32, dim=300):
    tokens = sentence.split()
    seq = np.zeros((max_len, dim), dtype=np.float32)
    j = 0
    for w in tokens:
        if w in ft_model:
            seq[j] = ft_model[w]
        else:
            # OOV는 0벡터로 둠(이미 0이라 생략 가능)
            pass
        j += 1
        if j >= max_len:
            break
    return seq

def build_sequences(sentences_list, indices, ft_model, max_len=32, dim=300):
    X = np.zeros((len(indices), max_len, dim), dtype=np.float32)
    for i, idx in enumerate(indices):
        X[i] = sentence_to_seq(sentences_list[idx], ft_model, max_len=max_len, dim=dim)
    return X

print("\n========== Building (32,300) Sequences ==========")
X_train_seq = build_sequences(sentences, X_train, ft_model=ft, max_len=MAX_LEN, dim=EMB_DIM)
X_val_seq   = build_sequences(sentences, X_val,   ft_model=ft, max_len=MAX_LEN, dim=EMB_DIM)
X_test_seq  = build_sequences(sentences, X_test,  ft_model=ft, max_len=MAX_LEN, dim=EMB_DIM)

print("X_train_seq:", X_train_seq.shape, "X_val_seq:", X_val_seq.shape, "X_test_seq:", X_test_seq.shape)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SeqDataset(X_train_seq, y_train)
val_ds   = SeqDataset(X_val_seq,   y_val)
test_ds  = SeqDataset(X_test_seq,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

import os

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, num_layers=1, num_classes=3, dropout=0.3):
        super().__init__()
        # num_layers=1이면 LSTM 내부 dropout은 적용 안 됨 -> 밖에서 dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 32, 300)
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, B, hidden_dim) -> 마지막 layer의 hidden
        h_last = h_n[-1]  # (B, hidden_dim)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)  # (B, num_classes)
        return logits

def eval_loader(model, loader, loss_fn, device):
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            loss = loss_fn(logits, yb)

            bs = yb.size(0)
            total_loss += loss.item() * bs
            n += bs

            pred = torch.argmax(logits, dim=1)
            all_y.append(yb.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    avg_loss = total_loss / n
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1, y_true, y_pred

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
ensure_dir(OUTDIR)

device = get_device()
print("\n========== Training LSTM ==========")
print("Device:", device)
set_seed(SEED)

model = LSTMClassifier(input_dim=EMB_DIM, hidden_dim=128, num_layers=1, num_classes=3, dropout=0.3).to(device)

loss_fn = nn.CrossEntropyLoss(weight=class_weights_t.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 35  # >= 30 required
history = {
    "train_loss": [], "train_acc": [], "train_f1": [],
    "val_loss": [], "val_acc": [], "val_f1": [],
}

best_val_f1 = -1.0
best_path = os.path.join(OUTDIR, "best_lstm.pth")

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    n = 0
    all_y = []
    all_pred = []

    for Xb, yb in train_loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        n += bs

        pred = torch.argmax(logits, dim=1)
        all_y.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    train_loss = total_loss / n
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    train_acc = accuracy_score(y_true, y_pred)
    train_f1 = f1_score(y_true, y_pred, average="macro")

    val_loss, val_acc, val_f1, _, _ = eval_loader(model, val_loader, loss_fn, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["train_f1"].append(train_f1)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_path)

    if epoch in [1, 5, 10, 20, 30, 35] or epoch == num_epochs:
        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | "
              f"best val f1 {best_val_f1:.4f}")

print("\n========== Evaluating Best LSTM on Test ==========")
model.load_state_dict(torch.load(best_path, map_location=device))
test_loss, test_acc, test_f1, y_true_test, y_pred_test = eval_loader(model, test_loader, loss_fn, device)

print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f} macroF1 {test_f1:.4f}")

cm = confusion_matrix(y_true_test, y_pred_test, labels=[0,1,2])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["neg","neu","pos"],
            yticklabels=["neg","neu","pos"])
plt.title("LSTM Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "lstm_confusion_matrix.png"))
plt.close()

def plot_curve(train_vals, val_vals, title, ylabel, filename):
    plt.figure()
    plt.plot(range(1, len(train_vals)+1), train_vals, label="train")
    plt.plot(range(1, len(val_vals)+1), val_vals, label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename))
    plt.close()

plot_curve(history["train_loss"], history["val_loss"], "LSTM Loss", "loss", "lstm_loss_learning_curve.png")
plot_curve(history["train_acc"],  history["val_acc"],  "LSTM Accuracy", "accuracy", "lstm_accuracy_learning_curve.png")
plot_curve(history["train_f1"],   history["val_f1"],   "LSTM Macro F1", "macro f1", "lstm_f1_learning_curve.png")

print("\nSaved to:", OUTDIR)
print(" - best_lstm.pth")
print(" - lstm_*_learning_curve.png")
print(" - lstm_confusion_matrix.png")
