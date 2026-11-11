"""Plot Part 1 training and validation losses at 100-step intervals."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    part1 = (
        (folder / "record.txt")
        .read_text()
        .split("Part 1:", 1)[1]
        .split("Part 2:", 1)[0]
    )
    configs = re.findall(
        r"Config ([AB]):(.*?)(?=Config [AB]:|\Z)", part1, flags=re.S
    )
    assert [name for name, _ in configs] == ["A", "B"]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 5.5), sharey=True, layout="constrained"
    )
    fig.suptitle("Training and validation loss vs steps", fontsize=20)
    for ax, (name, block) in zip(axes, configs):
        points = [
            (int(step), float(train_loss), float(val_loss))
            for step, train_loss, val_loss in re.findall(
                r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)",
                block,
                flags=re.M,
            )
            if int(step) % 100 == 0
        ]
        assert [step for step, _, _ in points] == list(range(0, 2000, 100))
        steps, train_losses, val_losses = zip(*points)

        ax.plot(
            steps, train_losses, color="#e67e22", marker="s", markersize=4,
            linewidth=2, label="Training loss"
        )
        ax.plot(
            steps, val_losses, color="#2563eb", marker="o", markersize=4,
            linewidth=2, label="Validation loss"
        )
        ax.set_title(f"Config {name}", fontsize=18, pad=12)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_xticks(range(0, 2000, 100))
        ax.tick_params(axis="x", labelrotation=45, labelsize=9)
        ax.set_ylim(1.6, 4.6)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()

    output = folder / "part1_val_loss.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"{output}: Config A and Config B, steps 0–1900 at 100-step intervals")


if __name__ == "__main__":
    main()
