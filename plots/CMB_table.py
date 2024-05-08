import NormalizingFlow as nf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import corner
import makedf

root = os.path.abspath("/home/rugved/Files/LuSEE/normalizingflows/")

##---------------------------------------------------------------------------##

def weighted_quantile(values, quantiles, sample_weight=None):
    return corner.core.quantile(values, quantiles, weights=sample_weight)


def plot_weighted_boxplot(samples, likelihoods, ax, y_value, c="C0", box_width=0.1):
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
            alpha=0.5,
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
# snrpps = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
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
chromaticerrbars = {"fmt": "d", "markersize": 5, "capsize": 4}
errbars95 = {"fmt": "", "capsize": 4, "xuplims": True, "markersize": 1}
dasnrpps = [1e4, 1e5, 1e6, 1e7]
cdsnrpps = [1e4, 1e5, 1e6, 1e7]
truth = 272.5
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
    4,
    1,
    figsize=(6, 8),
    sharex=True,
    gridspec_kw=dict(height_ratios=[0.5, 2, 0.5, 2]),
)
fig.subplots_adjust(hspace=0.1, bottom=0.15, top=0.95, left=0.15, right=0.85)

args = nf.Args()
args.fgFITS, args.freqs = "gsm16.fits", "51 101"
args.appendLik = "_cmb"
args.noiseSeed = 1

# noiseless
args.SNRpp = 1e24
for ics, cs in enumerate(combineSigmas):
    # achromatic
    args.combineSigma = cs
    args.chromatic = False
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= truth
    ax[0].axvline(truth, **truthaxvlines)
    plot_weighted_boxplot(s[:,0], nf.exp(ll), ax[0], 1 - 0.05 * ics, c=csColors[cs])
    # ax[0].errorbar(
    #     q50, 1 - 0.05 * ics, xerr=[[qm], [qp]], color=csColors[cs], **achromaticerrbars
    # )
    # ax[0].plot([0.1,q95], [1 - 0.05 * ics, 1 - 0.05 * ics], color=csColors[cs], lw=0.1)
    # ax[0].scatter([q95], [1 - 0.05 * ics], color=csColors[cs], marker="<", s=10)
    # ax[0].errorbar(
    #         q95, 1 - 0.05 * ics, xerr=[[1.0], [0]], color=csColors[cs], **errbars95
    #     )


    # chromatic
    args.chromatic = True
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= truth
    plot_weighted_boxplot(s[:,0], nf.exp(ll), ax[0], 0.75 - 0.05 * ics, c=csColors[cs])
    # q32, q50, q68, q95 = corner.core.quantile(
    #     s[:, 0], [0.32, 0.5, 0.68, 0.95], weights=nf.exp(ll)
    # )
    # qm, qp = q50 - q32, q68 - q50
    # ax[0].errorbar(
    #     q50,
    #     0.75 - 0.05 * ics,
    #     xerr=[[qm], [qp]],
    #     color=csColors[cs],
    #     **chromaticerrbars
    # )
    # ax[0].plot([0.1,q95], [0.75 - 0.05 * ics, 0.75 - 0.05 * ics], color=csColors[cs], lw=0.1)
    # ax[0].scatter([q95], [0.75 - 0.05 * ics], color=csColors[cs], marker="<", s=10)
    # ax[0].errorbar(
    #         q95, 0.75 - 0.05 * ics, xerr=[[1.0], [0]], color=csColors[cs], **errbars95
    #     )

# noisy
for isnrpp, snrpp in enumerate(dasnrpps):
    args.SNRpp = snrpp
    for ics, cs in enumerate(combineSigmas):
        # achromatic
        args.combineSigma = cs
        args.chromatic = False
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= truth
        ax[1].axvline(truth, **truthaxvlines)
        plot_weighted_boxplot(s[:,0], nf.exp(ll), ax[1], 1 + isnrpp - 0.05 * ics, c=csColors[cs])
        # q32, q50, q68, q95 = corner.core.quantile(
        #     s[:, 0], [0.32, 0.5, 0.68, 0.95], weights=nf.exp(ll)
        # )
        # qm, qp = q50 - q32, q68 - q50
        # ax[1].axvline(truth, **truthaxvlines)
        # ax[1].errorbar(
        #     q50,
        #     1 + isnrpp - 0.05 * ics,
        #     xerr=[[qm], [qp]],
        #     color=csColors[cs],
        #     **achromaticerrbars
        # )
        # ax[1].plot([0.1,q95], [1 + isnrpp - 0.05 * ics, 1 + isnrpp - 0.05 * ics], color=csColors[cs], lw=0.1)
        # ax[1].scatter([q95], [1 + isnrpp - 0.05 * ics], color=csColors[cs], marker="<", s=10)
        # ax[1].errorbar(
        #     q95, 1 + isnrpp - 0.05 * ics, xerr=[[1.0], [0]], color=csColors[cs], **errbars95
        # )

        # chromatic
        args.chromatic = True
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= truth
        plot_weighted_boxplot(s[:,0], nf.exp(ll), ax[1], 0.75 + isnrpp - 0.05 * ics, c=csColors[cs])
        # q32, q50, q68, q95 = corner.core.quantile(
        #     s[:, 0], [0.32, 0.5, 0.68, 0.95], weights=nf.exp(ll)
        # )
        # qm, qp = q50 - q32, q68 - q50
        # ax[1].errorbar(
        #     q50,
        #     0.75 + isnrpp - 0.05 * ics,
        #     xerr=[[qm], [qp]],
        #     color=csColors[cs],
        #     **chromaticerrbars
        # )
        # ax[1].plot([0.1,q95], [0.75 + isnrpp - 0.05 * ics, 0.75 + isnrpp - 0.05 * ics], color=csColors[cs], lw=0.1)
        # ax[1].scatter([q95], [0.75 + isnrpp - 0.05 * ics], color=csColors[cs], marker="<", s=10)
        # ax[1].errorbar(
        #     q95, 0.75 + isnrpp - 0.05 * ics, xerr=[[1.0], [0]], color=csColors[cs], **errbars95
        # )

##---------------------------------------------------------------------------##
# dark ages

args = nf.Args()
args.fgFITS, args.freqs, args.chromatic = "ulsa.fits", "1 51", False
args.appendLik = "_cmb"
args.noiseSeed = 1

# noiseless
args.SNRpp = 1e24
for ics, cs in enumerate(combineSigmas):
    # achromatic
    args.combineSigma = cs
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= truth
    ax[2].axvline(truth, **truthaxvlines)
    plot_weighted_boxplot(s[:,0], nf.exp(ll), ax[2], 1 - 0.05 * ics, c=csColors[cs])
    # q32, q50, q68, q95 = corner.core.quantile(
    #     s[:, 0], [0.32, 0.5, 0.68, 0.95], weights=nf.exp(ll)
    # )
    # qm, qp = q50 - q32, q68 - q50
    # ax[2].axvline(truth, **truthaxvlines)
    # ax[2].errorbar(
    #     q50, 1 - 0.05 * ics, xerr=[[qm], [qp]], color=csColors[cs], **achromaticerrbars
    # )
    # ax[2].plot([0.1,q95], [1 - 0.05 * ics, 1 - 0.05 * ics], color=csColors[cs], lw=0.1)
    # ax[2].scatter([q95], [1 - 0.05 * ics], color=csColors[cs], marker="<", s=10)
    # ax[2].errorbar(
    #         q95, 1 - 0.05 * ics, xerr=[[1.0], [0]], color=csColors[cs], **errbars95
    #     )

    # chromatic
    args.chromatic = True
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= truth
    plot_weighted_boxplot(s[:,0], nf.exp(ll), ax[2], 0.75 - 0.05 * ics, c=csColors[cs])
    # q32, q50, q68, q95 = corner.core.quantile(
    #     s[:, 0], [0.32, 0.5, 0.68, 0.95], weights=nf.exp(ll)
    # )
    # qm, qp = q50 - q32, q68 - q50
    # ax[2].errorbar(
    #     q50,
    #     0.75 - 0.05 * ics,
    #     xerr=[[qm], [qp]],
    #     color=csColors[cs],
    #     **chromaticerrbars
    # )
    # ax[2].plot([0.1,q95], [0.75 - 0.05 * ics, 0.75 - 0.05 * ics], color=csColors[cs], lw=0.1)
    # ax[2].scatter([q95], [0.75 - 0.05 * ics], color=csColors[cs], marker="<", s=10)
    # ax[2].errorbar(
    #         q95, 0.75 - 0.05 * ics, xerr=[[1.0], [0]], color=csColors[cs], **errbars95
    #     )

# noisy
for isnrpp, snrpp in enumerate(dasnrpps):
    args.SNRpp = snrpp
    for ics, cs in enumerate(combineSigmas):
        # achromatic
        args.combineSigma = cs
        args.chromatic = False
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= truth
        ax[3].axvline(truth, **truthaxvlines)
        plot_weighted_boxplot(s[:,0], nf.exp(ll), ax[3], 1 + isnrpp - 0.05 * ics, c=csColors[cs])
        # q32, q50, q68, q95 = corner.core.quantile(
        #     s[:, 0], [0.32, 0.5, 0.68, 0.95], weights=nf.exp(ll)
        # )
        # qm, qp = q50 - q32, q68 - q50
        # ax[3].axvline(truth, **truthaxvlines)
        # ax[3].errorbar(
        #     q50,
        #     1 + isnrpp - 0.05 * ics,
        #     xerr=[[qm], [qp]],
        #     color=csColors[cs],
        #     **achromaticerrbars
        # )
        # ax[3].plot([0.1,q95], [1 + isnrpp - 0.05 * ics, 1 + isnrpp - 0.05 * ics], color=csColors[cs], lw=0.1)
        # ax[3].scatter([q95], [1 + isnrpp - 0.05 * ics], color=csColors[cs], marker="<", s=10)
        # ax[3].errorbar(
        #     q95, 1 + isnrpp - 0.05 * ics, xerr=[[1.0], [0]], color=csColors[cs], **errbars95
        # )

        # chromatic
        args.chromatic = True
        s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        s[:, 0] *= truth
        plot_weighted_boxplot(s[:,0], nf.exp(ll), ax[3], 0.75 + isnrpp - 0.05 * ics, c=csColors[cs])
        # q32, q50, q68, q95 = corner.core.quantile(
        #     s[:, 0], [0.32, 0.5, 0.68, 0.95], weights=nf.exp(ll)
        # )
        # qm, qp = q50 - q32, q68 - q50
        # ax[3].errorbar(
        #     q50,
        #     0.75 + isnrpp - 0.05 * ics,
        #     xerr=[[qm], [qp]],
        #     color=csColors[cs],
        #     **chromaticerrbars
        # )
        # ax[3].plot([0.1,q95], [0.75 + isnrpp - 0.05 * ics, 0.75 + isnrpp - 0.05 * ics], color=csColors[cs], lw=0.1)
        # ax[3].scatter([q95], [0.75 + isnrpp - 0.05 * ics], color=csColors[cs], marker="<", s=10)
        # ax[3].errorbar(
        #     q95, 0.75 + isnrpp - 0.05 * ics, xerr=[[1.0], [0]], color=csColors[cs], **errbars95
        # )

##---------------------------------------------------------------------------##

ax[0].set_yticks([1.0], labels=["Noiseless"], minor=False, fontsize="small")
ax[2].set_yticks([1.0], labels=["Noiseless"], minor=False, fontsize="small")
ax[1].set_yticks(
    np.arange(1, len(dasnrpps) + 1),
    labels=[snrpplabels[snrpp] for snrpp in dasnrpps],
    minor=False,
)
ax[3].set_yticks(
    np.arange(1, len(cdsnrpps) + 1),
    labels=[snrpplabels[snrpp] for snrpp in cdsnrpps],
    minor=False,
)

ax[0].spines["bottom"].set_visible(False)
ax[2].spines["bottom"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[3].spines["top"].set_visible(False)

ax[0].xaxis.tick_top()
ax[2].xaxis.tick_top()

ax[3].set_xlabel(r"$A$ [mK]")

# ax[3].set_xlim(-0.5, 30)

ax[3].set_xlim(10,30000)
ax[3].set_xscale("log")

ax[0].axvspan(-0.5, 0.0, color="k", alpha=0.1)
ax[1].axvspan(-0.5, 0.0, color="k", alpha=0.1)
ax[2].axvspan(-0.5, 0.0, color="k", alpha=0.1)
ax[3].axvspan(-0.5, 0.0, color="k", alpha=0.1)

ax[1].set_ylabel("Signal-to-Nose Ratio", fontsize="small", labelpad=10)
ax[3].set_ylabel("Signal-to-Nose Ratio", fontsize="small", labelpad=10)

fig.text(0.9, 0.7, "Cosmic Dawn", rotation=270, fontsize="medium")
fig.text(0.9, 0.3, "Dark Ages", rotation=270, fontsize="medium")
fig.suptitle("CMB Monopole Likelihood", fontsize="large")

plt.plot([], [], **truthaxvlines, label="Truth")
plt.errorbar(
    [], [], xerr=[[], []], c=csColors[""], label=combineSigmalabels[""], lw=4.0
)
plt.errorbar(
    [],
    [],
    xerr=[[], []],
    c="gray",
    **achromaticerrbars,
    markerfacecolor="none",
    label="Achromatic"
)
plt.errorbar(
    [], [], xerr=[[], []], c=csColors["4"], label=combineSigmalabels["4"], lw=4.0
)
plt.errorbar(
    [],
    [],
    xerr=[[], []],
    c="gray",
    **chromaticerrbars,
    markerfacecolor="none",
    label="Chromatic"
)
plt.errorbar(
    [], [], xerr=[[], []], c=csColors["4 6"], label=combineSigmalabels["4 6"], lw=4.0
)
fig.legend(
    bbox_to_anchor=(0.5, 0.05), loc="center", borderaxespad=0, ncol=3, fontsize="small"
)

print("saving")
plt.savefig(f"CMB_table.pdf", bbox_inches="tight", dpi=300)

plt.show()
