import numpy as np
import NormalizingFlow as nf
import matplotlib.pyplot as plt
import corner


##---------------------------------------------------------------------------##


def weighted_quantile(values, quantiles, sample_weight=None):
    return corner.core.quantile(values, quantiles, weights=sample_weight)


def plot_weighted_boxplot(
    samples, likelihoods, ax, y_value, c="C0", hatch=None, box_width=0.1
):
    """Plot a horizontal weighted box plot on an Axis object."""
    # Calculate the weighted quantiles for 1-sigma, median, and 95% confidence interval
    sigma1 = weighted_quantile(samples, [0.159, 0.841], sample_weight=likelihoods)
    median = weighted_quantile(samples, [0.5], sample_weight=likelihoods)
    ci95 = weighted_quantile(samples, [0.025, 0.975], sample_weight=likelihoods)

    # Width of the boxplot, centered at y_value
    y_box = [y_value - box_width / 2, y_value + box_width / 2]

    # Plot the box (1-sigma)
    ax.add_patch(
        plt.Rectangle(
            (sigma1[0], y_value - box_width / 2),
            sigma1[1] - sigma1[0],
            box_width,
            facecolor=c,
            edgecolor=c,
            alpha=0.4,
            hatch=hatch,
        )
    )

    # Plot the whiskers (95% CI)
    ax.plot([ci95[0], ci95[0]], y_box, color=c)
    ax.plot([ci95[1], ci95[1]], y_box, color=c)

    # Plot the median
    ax.plot([median, median], y_box, color=c, marker="", markersize=8)

    # Plot the central line between whiskers
    ax.plot([ci95[0], ci95[1]], [y_value, y_value], color=c, linestyle="-", linewidth=1)


##---------------------------------------------------------------------------##

combineSigmas = ["", "4", "4 6"]
csColors = {"": "dimgray", "4": "C2", "4 6": "C3"}
combineSigmalabels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
truthaxvlines = {"c": "k", "ls": "--", "lw": 0.7}
cdtruth = 130
datruth = 40
daxlims = (15, 55)
cdxlims = (125, 135)
galcuts = [0.1, 10.0, 20.0, 30.0, 40.0]
gclabels = {
    0.1: r"$0.1^\circ$",
    10.0: r"$10.0^\circ$",
    20.0: r"$20.0^\circ$",
    30.0: r"$30.0^\circ$",
    40.0: r"$40.0^\circ$",
}

##---------------------------------------------------------------------------##


fig, ax = plt.subplots( 1, 2, figsize=(10, 6))
ax = ax.reshape(1, 2)
fig.subplots_adjust(bottom=0.18, top=0.9, left=0.1, right=0.9, wspace=0.3)

##---------------------------------------------------------------------------##
# cosmic dawn

args = nf.Args()
args.fgFITS, args.freqs = "gsm16.fits", "51 101"
args.SNRpp = 1e24

for igc, gc in enumerate(galcuts):
    args.galcut = gc
    for ics, cs in enumerate(combineSigmas):
        args.combineSigma = cs
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= cdtruth
        plot_weighted_boxplot(
            s[:, 0],
            nf.exp(ll),
            ax[0, 0],
            1 + igc - 0.05 * ics,
            c=csColors[cs],
        )

##---------------------------------------------------------------------------##
# dark ages

args = nf.Args()
args.fgFITS, args.freqs, args.chromatic = "ulsa.fits", "1 51", False
args.SNRpp = 1e24

for igc, gc in enumerate(galcuts):
    args.galcut = gc
    for ics, cs in enumerate(combineSigmas):
        args.combineSigma = cs
        args.chromatic = False
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= datruth
        plot_weighted_boxplot(
            s[:, 0],
            nf.exp(ll),
            ax[0, 1],
            1 + igc - 0.05 * ics,
            c=csColors[cs],
        )

##---------------------------------------------------------------------------##

ax[0, 0].set_yticks(
    np.arange(1, len(galcuts) + 1),
    labels=[gclabels[gc] for gc in galcuts],
    minor=False,
)
ax[0, 1].set_yticks(
    np.arange(1, len(galcuts) + 1),
    labels=[gclabels[gc] for gc in galcuts],
    minor=False,
)

ax[0, 0].axvline(cdtruth, **truthaxvlines)
ax[0, 1].axvline(datruth, **truthaxvlines)

ax[0, 0].set_xlabel(r"$A$ [mK]")
ax[0, 1].set_xlabel(r"$A$ [mK]")

# ax[0, 0].set_xlim(*cdxlims)
# ax[0, 1].set_xlim(*daxlims)

ax[0, 0].set_ylabel("Galaxy Cut", fontsize="small", labelpad=10)
ax[0, 1].set_ylabel("Galaxy Cut", fontsize="small", labelpad=10)

ax[0, 0].set_title("Cosmic Dawn", fontsize="medium")
ax[0, 1].set_title("Dark Ages", fontsize="medium")
fig.suptitle("Effect of Galaxy Cuts")

plt.plot([], [], **truthaxvlines, label="Truth")
plt.errorbar(
    [], [], xerr=[[], []], c=csColors[""], label=combineSigmalabels[""], lw=4.0
)
plt.errorbar(
    [], [], xerr=[[], []], c=csColors["4"], label=combineSigmalabels["4"], lw=4.0
)
plt.errorbar(
    [], [], xerr=[[], []], c=csColors["4 6"], label=combineSigmalabels["4 6"], lw=4.0
)
fig.legend(
    bbox_to_anchor=(0.5, 0.05), loc="center", borderaxespad=0, ncol=4, fontsize="small"
)

print("saving")
plt.savefig(f"galcut.pdf", bbox_inches="tight", dpi=300)

plt.show()
