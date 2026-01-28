import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
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

# data keys
word2idx = data["word2idx"]
idx2word = data["idx2word"]
counter = data["counter"]
skipgram_df = data["skipgram_df"]
vocab_size = len(word2idx)

# Precompute negative sampling distribution below
def build_neg_sampling_probs(counter, word2idx, power=0.75):
    counts = torch.zeros(len(word2idx), dtype=torch.float64)

    for w, c in counter.items():
        if w in word2idx:
            counts[word2idx[w]] = c
    
    probs = counts.pow(power)
    probs = probs / probs.sum()
    return probs.to(dtype=torch.float32)  # torch.multinomial이 foat이면 OK

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
    num_workers=2,
    pin_memory=(device.type == "cuda")
)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def make_targets(center, context, vocab_size):
    """
    center:  LongTensor, shape (B,)
    context: LongTensor, shape (B,)  # positive context
    vocab_size: int

    returns
      all_contexts: LongTensor, shape (B, 1+K)  # [pos | neg1..negK]
      labels:       FloatTensor, shape (B, 1+K) # [1, 0, ..., 0]
    """
    # center는 여기서 직접 쓰진 않지만, 함수 시그니처 맞추기 위해 받음
    device = center.device
    B = center.size(0)
    K = NEGATIVE_SAMPLES

    # 1) positive contexts: (B,1)
    pos = context.view(B, 1)

    # 2) negative sampling (CPU에서 뽑고 -> device로 이동)
    # neg_probs_cpu는 vocab_size 길이의 확률분포 텐서라고 가정
    assert neg_probs_cpu.numel() == vocab_size, "neg_probs_cpu size mismatch"

    neg = torch.multinomial(
        neg_probs_cpu, num_samples=B * K, replacement=True
    ).view(B, K)  # CPU tensor

    # 3) negative 안에 positive context가 섞이면 재샘플링해서 제거
    pos_cpu = pos.cpu()
    mask = (neg == pos_cpu)
    while mask.any():
        n_bad = int(mask.sum().item())
        neg[mask] = torch.multinomial(neg_probs_cpu, n_bad, replacement=True)
        mask = (neg == pos_cpu)

    # 4) device로 이동
    neg = neg.to(device, non_blocking=True)
    pos = pos.to(device, non_blocking=True)

    # 5) contexts 합치기: (B, 1+K)
    all_contexts = torch.cat([pos, neg], dim=1)

    # 6) labels 만들기: pos=1, neg=0  (BCEWithLogitsLoss용)
    labels = torch.zeros(B, 1 + K, device=device, dtype=torch.float32)
    labels[:, 0] = 1.0

    return all_contexts, labels

# Training loop
model.train()

for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0

    for step, (centers, contexts) in enumerate(loader, start=1):
        # 1) Move data to device (GPU if available)
        centers = centers.to(device, non_blocking=True)    # (B,)
        contexts = contexts.to(device, non_blocking=True)  # (B,)

        # 2) Build targets: sample negatives + create labels
        # all_contexts: (B, 1+K) = [positive | negatives]
        # labels:       (B, 1+K) = [1, 0, ..., 0]
        all_contexts, labels = make_targets(centers, contexts, vocab_size)

        # 3) Compute logits for (center, each context in all_contexts)
        #    dot product between center embedding and each context embedding
        v = model.in_embed(centers)            # (B, D)
        u = model.out_embed(all_contexts)      # (B, 1+K, D)

        logits = torch.bmm(u, v.unsqueeze(2)).squeeze(2)   # (B, 1+K)

        # 4) BCEWithLogitsLoss combines positive and negative loss automatically
        loss = bce(logits, labels)

        # 5) Backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # (선택) 로그
        if step % 2000 == 0:
            print(f"Epoch {epoch} Step {step}/{len(loader)}  avg_loss={running_loss/step:.4f}")

    print(f"Epoch {epoch} done. avg_loss={running_loss/len(loader):.4f}")


# Save embeddings and mappings
embeddings = model.in_embed.weight.detach().cpu().numpy()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
