import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SMOOTHING   = 1
TEMPERATURE = 0.7
NUM_SAMPLES = 10
SEED        = 9663

df         = pd.read_csv('names.txt', header=None, names=['name'])
df['name'] = df['name'].str.strip().str.lower()
words      = df['name'].tolist()
words_set  = set(words)

chars      = ['.'] + sorted(set(''.join(words)))
stoi       = {s:i for i,s in enumerate(chars)}
itos       = {i:s for s,i in stoi.items()}
vocab_size = len(chars)

N = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.int32)
for w in words:
    chs = [stoi[c] for c in ['.'] + list(w) + ['.']]
    for ix1, ix2, ix3 in zip(chs, chs[1:], chs[2:]):
        N[ix1, ix2, ix3] += 1

P = (N + SMOOTHING).astype(np.float32)
P /= P.sum(axis=2, keepdims=True)

def sample_with_temp(p_row, rng, temperature=TEMPERATURE):
    logits = np.log(p_row + 1e-10)
    logits /= temperature
    probs   = np.exp(logits - logits.max())
    probs  /= probs.sum()
    return rng.choice(len(probs), p=probs)

rng       = np.random.default_rng(seed=SEED)
generated = []

while len(generated) < NUM_SAMPLES:
    out = []
    ix1, ix2 = 0, 0
    while True:
        ix3 = sample_with_temp(P[ix1, ix2], rng)
        if ix3 == 0:
            break
        out.append(itos[ix3])
        ix1, ix2 = ix2, ix3
    name = ''.join(out)
    if name and name not in words_set:
        generated.append(name.capitalize())

for name in generated:
    print(name)


bigram_counts = N.sum(axis=2)
top_bigrams   = np.argsort(bigram_counts.flatten())[-12:][::-1]
top_pairs     = [(idx // vocab_size, idx % vocab_size) for idx in top_bigrams]


char_freq   = N.sum(axis=(0, 1))
top_chars   = np.argsort(char_freq)[-15:][::-1]
top_chars   = sorted(top_chars)         

row_labels  = [f"{itos[i1]}{itos[i2]}" for i1, i2 in top_pairs]
col_labels  = [itos[c] for c in top_chars]
heat_data   = np.array([[P[i1, i2, c] for c in top_chars] for i1, i2 in top_pairs])

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(heat_data, aspect='auto', cmap='Blues')

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=11)
ax.set_xlabel("Next character", fontsize=12)
ax.set_ylabel("Bigram context", fontsize=12)
ax.set_title("Trigram probabilities: P(next char | bigram context)", fontsize=13)

plt.colorbar(im, ax=ax, label="Probability")
plt.tight_layout()
plt.savefig("trigram_heatmap.png", dpi=150)
plt.show()