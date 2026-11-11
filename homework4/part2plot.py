"""Compare baseline and Part 2 validation losses at 100-step intervals."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def validation_points(block):
    """Read validation losses, keeping only steps divisible by 100."""
    points = [
        (int(step), float(loss))
        for step, loss in re.findall(
            r"^step (\d+): train loss [\d.]+, val loss ([\d.]+)",
            block,
            flags=re.M,
        )
        if int(step) % 100 == 0
    ]
    assert [step for step, _ in points] == list(range(0, 2000, 100))
    return zip(*points)


def main():
    folder = Path(__file__).resolve().parent
    record = (folder / "record.txt").read_text()
    part1, part2 = record.split("Part 2:", 1)
    baseline = part1.split("Config A:", 1)[1].split("Config B:", 1)[0]
    variants = dict(
        re.findall(r"^([abc]):[^\n]*\n(.*?)(?=^[abc]:|\Z)", part2, flags=re.M | re.S)
    )
    curves = [
        ("baseline", baseline, "#2563eb", "-", 3),
        ("no causal mask", variants["a"], "#e67e22", "-", 2),
        ("no positional embedding", variants["b"], "#159447", "--", 2),
        ("one head", variants["c"], "#9333ea", ":", 2),
    ]

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for label, block, color, style, width in curves:
        steps, losses = validation_points(block)
        ax.plot(
            steps,
            losses,
            label=label,
            color=color,
            linestyle=style,
            linewidth=width,
        )

    ax.set_title("Validation Loss", fontsize=18, pad=12)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Validation loss", fontsize=12)
    ax.set_xticks(range(0, 2000, 100))
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.set_ylim(0, 4.6)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()

    output = folder / "part2_val_loss.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"{output}: four curves, steps 0–1900 at 100-step intervals")


if __name__ == "__main__":
    main()
