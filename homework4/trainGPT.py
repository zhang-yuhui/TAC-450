import argparse
import time
import torch
from GPTmodel import Transformer

parser = argparse.ArgumentParser(description='Train a character-level GPT model.')
parser.add_argument(
    '--save-model',
    nargs='?',
    const='model.pt',
    default=None,
    metavar='PATH',
    help='Save a checkpoint after training (default path: model.pt).',
)
args = parser.parse_args()

torch.manual_seed(42)

# Training hyperparameters
batch_size = 16
seq_size = 16
embed_dim = 128
n_heads = 1
n_layers = 4
dropout = 0.0
learning_rate = 0.0005
n_epoch = 2000
print_interval = 100
eval_iters = 200
inference_token = 300
masked = True
positional_embedding = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Character tokenization and a 90/10 training/validation split.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

c2i = {c:i for i, c in enumerate(chars)} # character to integer
i2c = {i:c for i, c in enumerate(chars)} # integer to character

# encode a string to a "list of integers"
encode=lambda s: [c2i[c] for c in s]
# decode a list of integers to a string
decode=lambda l: ''.join([i2c[i] for i in l])

data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_ind = torch.randint(0, len(data) - seq_size, (batch_size,))
    context = torch.stack([data[i:i + seq_size] for i in start_ind])
    target = torch.stack([data[i + 1:i + seq_size + 1] for i in start_ind])
    return context, target


@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            context, target = get_batch(split)
            _, loss = model(context, target)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = Transformer(
    vocab_size=vocab_size,
    seq_size=seq_size,
    embed_dim=embed_dim,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
    masked=masked,
    positional_embedding=positional_embedding
).to(device)

print(f'The model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training
start = time.time()
for epoch in range(n_epoch):
    context, target = get_batch('train')
    _, loss = model(context, target)

    if epoch % print_interval == 0 or epoch == n_epoch - 1:
        losses = estimate_loss(model, eval_iters)
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f'Training took {time.time() - start:.2f} seconds')

if args.save_model is not None:
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'vocab_size': vocab_size,
                'seq_size': seq_size,
                'embed_dim': embed_dim,
                'n_heads': n_heads,
                'n_layers': n_layers,
                'dropout': dropout,
                'masked' : masked,
                'positional_embedding': positional_embedding
            },
            'chars': chars,
        },
        args.save_model,
    )
    print(f'Model saved to {args.save_model}')

# Inference
model.eval()
start=time.time()
context  = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.inference_mode():
    response = model.generation(context, max_tokens=inference_token)[0].tolist()
print(f'Inference took {time.time()-start} seconds')
print('---')
print(decode(response))
