import NormalizingFlow as nf
import numpy as np
import matplotlib.pyplot as plt
import corner

##---------------------------------------------------------------------------##


def weighted_quantile(values, quantiles, sample_weight=None):
    return corner.core.quantile(values, quantiles, weights=sample_weight)


def plot_weighted_boxplot(
    samples, likelihoods, ax, y_value, c="C0", box_width=0.1, **kwargs
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
            **kwargs
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
# plot params

colors = {"": "gray", "4": "C2", "4 6": "C3"}
lstyles = ["-.", ":", "-"]
labels = {
    "": r"$2^\circ$",
    "4": r"$2^\circ+4^\circ$",
    "4 6": r"$2^\circ+4^\circ+6^\circ$",
}
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
combineSigmas = ["", "4", "4 6"]
csColors = {"": "dimgray", "4": "C2", "4 6": "C3"}
combineSigmalabels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
sigmaLabels = {2: "$2^\circ$", 4: "$4^\circ$", 6: "$6^\circ$"}
truthaxvlines = {"c": "k", "ls": "--", "lw": 0.7}
achromaticerrbars = {"fmt": "o", "markersize": 5, "capsize": 4}
chromaticerrbars = {"ls":"--", "hatch":"////", "rasterized":True}
errbars95 = {"fmt": "", "capsize": 4, "xuplims": True, "markersize": 1}
dasnrpps = [1e8, 1e9, 1e10, 1e11]
cdsnrpps = [1e5, 1e6, 1e7, 1e8]
datruth = 40
cdtruth = 130
snrpplabels = {
    1e4: r"$10^4$",
    1e5: r"$10^5$",
    1e6: r"$10^6$",
    1e7: r"$10^7$",
    1e8: r"$10^8$",
    1e9: r"$10^9$",
    1e10: r"$10^{10}$",
    1e11: r"$10^{11}$",
    1e12: r"$10^{12}$",
    1e13: r"$10^{13}$",
    1e24: r"$10^{24}$",
}

##---------------------------------------------------------------------------##

fig, ax = plt.subplots(
    2,
    2,
    figsize=(10, 6),
    sharex="col",
    gridspec_kw=dict(height_ratios=[0.4, 2]),
)
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.15, right=0.85, wspace=0.3)

args = nf.Args()
args.fgFITS, args.freqs = "gsm16.fits", "51 101"
args.appendLik = "_1D"

# noiseless
args.SNRpp = 1e24
for ics, cs in enumerate(combineSigmas):
    # achromatic
    args.combineSigma = cs
    args.chromatic = False
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= cdtruth
    plot_weighted_boxplot(
        s[:, 0],
        nf.exp(ll),
        ax[0, 0],
        1 - 0.05 * ics,
        c=csColors[cs],
    )

    # chromatic
    args.chromatic = True
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= cdtruth
    plot_weighted_boxplot(
        s[:, 0],
        nf.exp(ll),
        ax[0, 0],
        0.75 - 0.05 * ics,
        c=csColors[cs],
        **chromaticerrbars,
    )

# noisy
for isnrpp, snrpp in enumerate(cdsnrpps):
    args.SNRpp = snrpp
    for ics, cs in enumerate(combineSigmas):
        # achromatic
        args.combineSigma = cs
        args.chromatic = False
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= cdtruth
        plot_weighted_boxplot(
            s[:, 0],
            nf.exp(ll),
            ax[1, 0],
            1 + isnrpp - 0.05 * ics,
            c=csColors[cs],
        )

        # chromatic
        args.chromatic = True
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= cdtruth
        plot_weighted_boxplot(
            s[:, 0],
            nf.exp(ll),
            ax[1, 0],
            0.75 + isnrpp - 0.05 * ics,
            c=csColors[cs],
            **chromaticerrbars,
        )

##---------------------------------------------------------------------------##
# dark ages

args = nf.Args()
args.fgFITS, args.freqs, args.chromatic = "ulsa.fits", "1 51", False
args.appendLik = "_1D"
args.noiseSeed = 0

# noiseless
args.SNRpp = 1e24
for ics, cs in enumerate(combineSigmas):
    args.combineSigma = cs
    # achromatic
    args.chromatic = False
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= datruth
    plot_weighted_boxplot(
        s[:, 0],
        nf.exp(ll),
        ax[0, 1],
        1 - 0.05 * ics,
        c=csColors[cs],
    )

    # chromatic
    args.chromatic = True
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= datruth
    plot_weighted_boxplot(
        s[:, 0],
        nf.exp(ll),
        ax[0, 1],
        0.75 - 0.05 * ics,
        c=csColors[cs],
        **chromaticerrbars
    )

# noisy
for isnrpp, snrpp in enumerate(dasnrpps):
    args.SNRpp = snrpp
    for ics, cs in enumerate(combineSigmas):
        # achromatic
        args.combineSigma = cs
        args.chromatic = False
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= datruth
        plot_weighted_boxplot(
            s[:, 0],
            nf.exp(ll),
            ax[1, 1],
            1 + isnrpp - 0.05 * ics,
            c=csColors[cs],
        )

        # chromatic
        args.chromatic = True
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= datruth
        plot_weighted_boxplot(
            s[:, 0],
            nf.exp(ll),
            ax[1, 1],
            0.75 + isnrpp - 0.05 * ics,
            c=csColors[cs],
            **chromaticerrbars,
        )

##---------------------------------------------------------------------------##

ax[0, 0].set_yticks([1.0], labels=["Noiseless"], minor=False, fontsize="small")
ax[0, 1].set_yticks([1.0], labels=["Noiseless"], minor=False, fontsize="small")

ax[1, 0].set_yticks(
    np.arange(1, len(cdsnrpps) + 1),
    labels=[snrpplabels[snrpp] for snrpp in cdsnrpps],
    minor=False,
)
ax[1, 1].set_yticks(
    np.arange(1, len(dasnrpps) + 1),
    labels=[snrpplabels[snrpp] for snrpp in dasnrpps],
    minor=False,
)

ax[0, 0].set_ylim(0.3, 1.2)
ax[0, 1].set_ylim(0.3, 1.2)
ax[1, 0].set_ylim(0.3, len(cdsnrpps) + 0.2)
ax[1, 1].set_ylim(0.3, len(dasnrpps) + 0.2)

ax[0, 0].axvline(cdtruth, **truthaxvlines)
ax[1, 0].axvline(cdtruth, **truthaxvlines)
ax[0, 1].axvline(datruth, **truthaxvlines)
ax[1, 1].axvline(datruth, **truthaxvlines)

ax[0, 0].axvspan(1e-3, 1e-2 * cdtruth, color="k", alpha=0.1)
ax[1, 0].axvspan(1e-3, 1e-2 * cdtruth, color="k", alpha=0.1)
ax[0, 1].axvspan(1e-3, 1e-2 * datruth, color="k", alpha=0.1)
ax[1, 1].axvspan(1e-3, 1e-2 * datruth, color="k", alpha=0.1)

ax[0, 0].spines["bottom"].set_visible(False)
ax[0, 1].spines["bottom"].set_visible(False)
ax[1, 0].spines["top"].set_visible(False)
ax[1, 1].spines["top"].set_visible(False)

ax[0, 0].xaxis.tick_top()
ax[0, 1].xaxis.tick_top()

ax[1, 0].set_xlabel(r"$A$ [mK]")
ax[1, 1].set_xlabel(r"$A$ [mK]")

ax[1, 0].set_xlim(0.1, 30000)
ax[1, 1].set_xlim(0.1, 30000)

ax[1, 0].set_xscale("log")
ax[1, 1].set_xscale("log")

ax[1, 0].set_ylabel("Signal-to-Nose Ratio", fontsize="small", labelpad=10)
ax[1, 1].set_ylabel("Signal-to-Nose Ratio", fontsize="small", labelpad=10)

ax[0, 0].set_title("Cosmic Dawn", fontsize="small")
ax[0, 1].set_title("Dark Ages", fontsize="small")
fig.suptitle("Signal Amplitude Likelihoods - Effect of Chromaticity", fontsize="medium")

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
plt.gca().add_patch(
    plt.Rectangle((0, 0), 0, 0, facecolor="none", edgecolor="gray", label="Achromatic")
)
plt.gca().add_patch(
    plt.Rectangle(
        (0, 0),
        0,
        0,
        facecolor="none",
        edgecolor="gray",
        **chromaticerrbars,
        label="Chromatic",
    )
)

h, l = plt.gca().get_legend_handles_labels()
reord = [0, 3, 1, 5, 2, 4]
fig.legend(
    [h[i] for i in reord],
    [l[i] for i in reord],
    bbox_to_anchor=(0.5, 0.05),
    loc="center",
    borderaxespad=0,
    ncol=3,
    fontsize="small",
)

print("saving")
plt.savefig(f"chromaticity.pdf", bbox_inches="tight", dpi=300)

plt.show()
