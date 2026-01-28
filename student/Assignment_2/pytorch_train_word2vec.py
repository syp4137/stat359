import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        # skipgram_df: pandas DataFrame with columns ['center', 'context']
        # NOTE: DataFrame 그대로 쓰면 느려서 tensor로 변환해 둠
        self.centers = torch.as_tensor(skipgram_df["center"].to_numpy(dtype="int64"))
        self.contexts = torch.as_tensor(skipgram_df["context"].to_numpy(dtype="int64"))

    def __len__(self):
        return self.centers.numel()

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward_pos(self, centers, contexts):
        # centers: (B,), contexts: (B,)
        v = self.in_embed(centers)      # (B, D)
        u = self.out_embed(contexts)    # (B, D)
        return (v * u).sum(dim=1)       # (B,)


# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

# data keys: sent_list, counter, word2idx, idx2word, skipgram_df
word2idx = data["word2idx"]
idx2word = data["idx2word"]
counter = data["counter"]
skipgram_df = data["skipgram_df"]

vocab_size = len(word2idx)

# Precompute negative sampling distribution below
def build_neg_sampling_probs(counter, word2idx, power=0.75):
    # counter: Counter(word -> count)
    # word2idx: dict(word -> idx)
    counts = torch.zeros(len(word2idx), dtype=torch.float64)
    for w, c in counter.items():
        if w in word2idx:
            counts[word2idx[w]] = c
    probs = counts.pow(power)
    probs = probs / probs.sum()
    return probs.to(dtype=torch.float32)  # torch.multinomial이 float이면 OK

neg_probs_cpu = build_neg_sampling_probs(counter, word2idx)  # CPU tensor

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,     # Colab에서 문제 생기면 0으로
    pin_memory=(device.type == "cuda")
)

# Model, Loss, Optimizer


def make_targets(center, context, vocab_size):
    pass

# Training loop
def resample_if_matches(neg, contexts, probs_cpu):
    # neg: (B, K), contexts: (B,)
    mask = (neg == contexts.unsqueeze(1))
    while mask.any():
        n = int(mask.sum().item())
        neg[mask] = torch.multinomial(probs_cpu, n, replacement=True).to(neg.device)
        mask = (neg == contexts.unsqueeze(1))
    return neg

model.train()
for epoch in range(1, EPOCHS + 1):
    running = 0.0

    for step, (centers, contexts) in enumerate(loader, start=1):
        centers = centers.to(device, non_blocking=True)   # (B,)
        contexts = contexts.to(device, non_blocking=True) # (B,)
        B = centers.size(0)

        # --- negative sampling (B, K) ---
        neg = torch.multinomial(
            neg_probs_cpu, num_samples=B * NEGATIVE_SAMPLES, replacement=True
        ).view(B, NEGATIVE_SAMPLES).to(device, non_blocking=True)

        # (권장) positive context가 negative에 섞이면 재샘플링
        neg = resample_if_matches(neg, contexts, neg_probs_cpu)

        # --- positive logits ---
        pos_logits = model.forward_pos(centers, contexts)     # (B,)
        pos_labels = torch.ones_like(pos_logits)

        # --- negative logits ---
        v_center = model.in_embed(centers)        # (B, D)
        u_neg = model.out_embed(neg)              # (B, K, D)
        neg_logits = torch.bmm(u_neg, v_center.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_labels = torch.zeros_like(neg_logits)

        # --- loss ---
        loss = bce(pos_logits, pos_labels) + bce(neg_logits, neg_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()

        if step % 2000 == 0:
            print(f"Epoch {epoch} Step {step}/{len(loader)} loss={running/step:.4f}")

    print(f"Epoch {epoch} done. avg loss={running/len(loader):.4f}")

# Save embeddings and mappings
embeddings = model.in_embed.weight.detach().cpu().numpy()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
