with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Try tiktoken later

# Simple mapping/tokenizer
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encode: take a string, output a list f integers
decode = lambda l: ''.join([itos[i] for i in l]) # decode: take a list of integers, output a string

print(encode("hi there"))
print(decode(encode("hi there")))

import torch


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")


print(torch.__version__)  # Should return 2.4.0
print(torch.cuda.is_available())  # Should return True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data = torch.tensor(encode(text), dtype=torch.long)
data = data.to(device) # Remove?

print(data.shape, data.dtype)
print(data[:1000])


n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


block_size= 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When inpuit is {context} the target is: {target}")


torch.manual_seed(1337)
batch_size = 4 # how many independent squences will we process in parallel?
block_size = 8 # What is the maximum context length for predictions?

def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # Random offset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocal_size):
        super().__init__()
        # each tocken directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # or -1
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of idices in the current context
        for _ in range(max_new_tokens):
            # Get the prdictions
            logits, loss = self(idx)
            # forcus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            #apply softmax to get probabilites
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
m.to(device) # Remove??
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)


idx = torch.zeros((1,1), dtype=torch.long)
idx = idx.to(device)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
    # sample a batch of data
    xb, yb = get_batch('train')

    #evalutate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
idx = torch.zeros((1,1), dtype=torch.long)
idx = idx.to(device)
print(decode(m.generate(idx, max_new_tokens=300)[0].tolist()))
