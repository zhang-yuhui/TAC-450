import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from GPTmodel import Transformer

MODEL_PATH = "baseline.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = Transformer(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

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

test_input = torch.tensor(encode(text[:16]))
test_input = test_input.reshape(1, -1)

weights_baseline = []
weights_nomask = []
with torch.inference_mode():
    model(test_input, None)
    weights_baseline = model.last_weights
    for i in range(4):
        weights_baseline[i] = weights_baseline[i].reshape(16, 16)

MODEL_PATH = "nomasked.pt"

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = Transformer(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

with torch.inference_mode():
    model(test_input, None)
    weights_nomask = model.last_weights
    for i in range(4):
        weights_nomask[i] = weights_nomask[i].reshape(16, 16)

def plot_attention_scores(list1, list2):
    if len(list1) != 4 or len(list2) != 4:
        raise ValueError("Each list must contain exactly 4 tensors.")

    # Detach from autograd and move GPU tensors to CPU for plotting.
    groups = [
        [tensor.detach().cpu().float().numpy() for tensor in tensors]
        for tensors in (list1, list2)
    ]

    if any(matrix.shape != (16, 16) for group in groups for matrix in group):
        raise ValueError("Each tensor must have shape (16, 16).")

    vmin = min(matrix.min() for group in groups for matrix in group)
    vmax = max(matrix.max() for group in groups for matrix in group)

    fig = plt.figure(figsize=(13, 6), layout="constrained")
    outer = fig.add_gridspec(1, 2, wspace=0.15)
    axes = []

    for group_idx, group in enumerate(groups):
        inner = outer[0, group_idx].subgridspec(
            3, 2, height_ratios=[0.15, 1, 1]
        )

        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis("off")
        title_ax.text(
            0.5, 0.5,
            ["Baseline", "No causal mask"][group_idx],
            ha="center", va="center",
            fontsize=16, fontweight="bold",
        )

        for tensor_idx, matrix in enumerate(group):
            row, col = divmod(tensor_idx, 2)
            ax = fig.add_subplot(inner[row + 1, col])

            im = ax.imshow(
                matrix,
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            for i in range(matrix.shape[0]):
                ax.add_patch(
                    Rectangle(
                        (i - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
            ax.set_title(f"Head {tensor_idx + 1}")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            ax.set_xticks([0, 5, 10, 15])
            ax.set_yticks([0, 5, 10, 15])
            axes.append(ax)

    fig.colorbar(im, ax=axes, label="Attention score", shrink=0.85)
    fig.savefig("attention_score_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


plot_attention_scores(weights_baseline, weights_nomask)