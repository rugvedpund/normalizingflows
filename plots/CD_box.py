# create a pretty plot to replace the table

import NormalizingFlow as nf
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.stats import gaussian_kde

args = nf.Args()
args.fgFITS, args.freqs = "gsm16.fits", "51 101"

npoints = (70, 71, 69)  # weird, but correct
truths = [130, 20, 67.5]
amin, amax = 60.0, 240.0
wmin, wmax = 17.0, 29.0
nmin, nmax = 65.0, 70.0
limits = [(0.01 * truths[0], 10 * truths[0]), (10, 40), (50, 90)]  # cube
labels = {"amp": r"$A$", "width": r"$\nu_{\rm rms}$", "numin": r"$\nu_{\rm min}$"}
chromatic = "Chromatic" if args.chromatic == "True" else "Achromatic"
sigmaLabels = {2: "$2^\circ$", 4: "$4^\circ$", 6: "$6^\circ$"}
csColors = {"": "dimgray", "4": "C2", "4 6": "C3"}
combineSigmalabels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
truthaxvlines = {"c": "k", "ls": ":", "lw": 0.7}
achromaticerrbars = {"fmt": "o", "markersize": 1, "capsize": 1}
chromaticerrbars = {"fmt": "d", "markersize": 1, "capsize": 1}
snrpps = [1e5, 1e6, 1e7, 1e8]
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
gainFlabels = {
    0.01: r"$1\%$",
    0.05: r"$5\%$",
}
freqFlabels = {
    0.01: r"$1\%$",
    0.05: r"$5\%$",
}

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

fig, ax = plt.subplots(
    4,
    3,
    figsize=(10, 7),
    sharex="col",
    gridspec_kw=dict(height_ratios=[0.5, 2,1,1]),
)
fig.subplots_adjust(
    left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.05, hspace=0.1
)

for ins, nseed in enumerate([0]):
    args.noiseSeed = nseed
    print(f"noiseSeed {nseed}")

    # for noiseless
    print("Noiseless")
    args.SNRpp = 1e24
    for ics, cs in enumerate(["", "4", "4 6"]):
        # achromatic
        args.combineSigma = cs
        args.appendLik = "_smallcube"
        args.chromatic = False
        s, ll = nf.get_samplesAndLikelihood(args, plot="all")
        try:
            s, ll = nf.get_samplesAndLikelihood(args, plot="all")
            assert ll.size == 342930
        except FileNotFoundError:
            print(f"SNRpp 1e24 chromatic {cs} {args.appendLik} not found")
            s, ll = np.zeros_like(s), np.zeros_like(ll)
            pass
        except AssertionError:
            print(ll.size)
            print(f"SNRpp 1e24 achromatic {cs} {args.appendLik} bad shape")
            pass

        s[:, 0] *= truths[0]
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[0, i].axvline(truths[i], **truthaxvlines)
            plot_weighted_boxplot(
                s[:, i],
                nf.exp(ll),
                ax[0, i],
                1 - 0.15* ics - 0.02 * ins,
                c=csColors[cs],
            )
            # ax[0,i].errorbar(
            #     constraints[p],
            #     1 - 0.05 * ics,
            #     xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
            #     **achromaticerrbars,
            #     c=csColors[cs],
            # )

        # # chromatic
        # args.chromatic = True
        # args.appendLik = "_cube"
        # try:
        #     s, ll = nf.get_samplesAndLikelihood(args, plot="all")
        #     assert ll.size == 342930
        # except FileNotFoundError:
        #     s,ll=np.zeros_like(s),np.zeros_like(ll)
        #     print(f"SNRpp 1e24 chromatic {cs} {args.appendLik} not found")
        #     pass
        # except AssertionError:
        #     print(ll.size)
        #     print(f"SNRpp 1e24 chromatic {cs} {args.appendLik} bad shape")
        #     pass
        # constraints = nf.get_constraints(s, ll)
        # constraints["amp"] *= truths[0]
        # constraints["amp+"] *= truths[0]
        # constraints["amp-"] *= truths[0]
        # for i, p in enumerate(["amp", "width", "numin"]):
        #     plot_weighted_boxplot(s[:,i],nf.exp(ll), ax[0,i], 0.75-0.05*ics)
        #     ax[0, i].errorbar(
        #         constraints[p],
        #         0.75 - 0.05 * ics,
        #         xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
        #         **chromaticerrbars,
        #         c=csColors[cs],
        #         # c="C" + str(ics),
        #     )

    ##---------------------------------------------------------------------------##

    print("Noisy")
    # for noisy
    for isnrpp, snrpp in enumerate(snrpps):
        for ics, cs in enumerate(["", "4", "4 6"]):
            # achromatic
            if args.SNRpp>=1e6:
                args.appendLik = '_smallcube'
            else:
                args.appendLik = "_cube"
            args.chromatic = False
            args.SNRpp = snrpp
            args.combineSigma = cs
            try:
                s, ll = nf.get_samplesAndLikelihood(args, plot="all")
                assert ll.size == 342930
            except FileNotFoundError:
                s, ll = np.zeros_like(s), np.zeros_like(ll)
                print(f"SNRpp {snrpp:.0e} achromatic {cs} {args.appendLik} not found")
                pass
            except AssertionError:
                print(ll.size)
                print(f"SNRpp {snrpp:.0e} achromatic {cs} {args.appendLik} bad shape")
                pass
            s[:, 0] *= truths[0]
            for i, p in enumerate(["amp", "width", "numin"]):
                ax[1, i].axvline(truths[i], **truthaxvlines)
                plot_weighted_boxplot(
                    s[:, i],
                    nf.exp(ll),
                    ax[1, i],
                    isnrpp + 1 - 0.15 * ics - 0.02 * ins,
                    c=csColors[cs],
                )
            # # plot
            # for i, p in enumerate(["amp", "width", "numin"]):
            #     ax[1, i].axvline(truths[i], **truthaxvlines)
            #     ax[1, i].errorbar(
            #         constraints[p],
            #         isnrpp + 1 - 0.05 * ics,
            #         xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
            #         **achromaticerrbars,
            #         c=csColors[cs],
            #     )

            # # # chromatic
            # args.chromatic = True
            # try:
            #     s, ll = nf.get_samplesAndLikelihood(args, plot="all")
            #     assert ll.size == 342930
            # except FileNotFoundError:
            #     s,ll=np.zeros_like(s),np.zeros_like(ll)
            #     print(f"SNRpp {snrpp:.0e} chromatic {cs} {args.appendLik} not found")
            #     pass
            # except AssertionError:
            #     print(ll.size)
            #     print(f"SNRpp {snrpp:.0e} achromatic {cs} {args.appendLik} bad shape")
            #     pass
            # constraints = nf.get_constraints(s, ll)
            # constraints["amp"] *= truths[0]
            # constraints["amp+"] *= truths[0]
            # constraints["amp-"] *= truths[0]
            # for i, p in enumerate(["amp", "width", "numin"]):
            #     ax[1, i].errorbar(
            #         constraints[p],
            #         isnrpp + 0.75 - 0.05 * ics,
            #         xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
            #         **chromaticerrbars,
            #         c=csColors[cs],
            #         # c="C" + str(ics),
            #     )

    ##---------------------------------------------------------------------------##

    print("gainF")
    for ics, cs in enumerate(["", "4", "4 6"]):
        for igF, gF in enumerate([0.01, 0.05]):
            args.SNRpp = 1e24
            args.gainFluctuationLevel = gF
            args.combineSigma = cs
            args.appendLik = "_smallcube"
            args.chromatic = False
            try:
                s, ll = nf.get_samplesAndLikelihood(args, plot="all")
                assert ll.size == 342930
            except FileNotFoundError:
                s, ll = np.zeros_like(s), np.zeros_like(ll)
                print(f"gainF achromatic {cs} {args.appendLik} not found")
                pass
            except AssertionError:
                print(ll.size)
                print(f"gainF achromatic {cs} {args.appendLik} bad shape")
                pass
            s[:, 0] *= truths[0]
            for i, p in enumerate(["amp", "width", "numin"]):
                ax[2, i].axvline(truths[i], **truthaxvlines)
                plot_weighted_boxplot(
                    s[:, i],
                    nf.exp(ll),
                    ax[2, i],
                    1 - igF - 0.15 * ics - 0.02 * ins,
                    c=csColors[cs],
                )
    args.gainFluctuationLevel = 0.0

    ##---------------------------------------------------------------------------##

    print("freqF")
    for ics, cs in enumerate(["", "4", "4 6"]):
        for ifF, fF in enumerate([0.01, 0.05]):
            args.SNRpp = 1e24
            args.freqFluctuationLevel = fF
            args.combineSigma = cs
            args.appendLik = "_smallcube"
            args.chromatic = False
            try:
                s, ll = nf.get_samplesAndLikelihood(args, plot="all")
                assert ll.size == 342930
            except FileNotFoundError:
                s, ll = np.zeros_like(s), np.zeros_like(ll)
                print(f"freqF achromatic {cs} {args.appendLik} not found")
                pass
            except AssertionError:
                print(ll.size)
                print(f"freqF achromatic {cs} {args.appendLik} bad shape")
                pass
            s[:, 0] *= truths[0]
            for i, p in enumerate(["amp", "width", "numin"]):
                ax[3, i].axvline(truths[i], **truthaxvlines)
                plot_weighted_boxplot(
                    s[:, i],
                    nf.exp(ll),
                    ax[3, i],
                    1 - ifF - 0.15 * ics - 0.02 * ins,
                    c=csColors[cs],
                )
    args.freqFluctuationLevel = 0.0


##---------------------------------------------------------------------------##
print("done")

ax[0, 0].set_yticks(
    [1.0],
    labels=["Noiseless"],
    minor=False,
)
ax[1, 0].set_yticks(
    np.arange(1, len(snrpps) + 1),
    labels=[snrpplabels[snrpp] for snrpp in snrpps],
    minor=False,
)

ax[2, 0].set_yticks(
    [1.0, 0.0],
    labels=[gainFlabels[gF] for gF in [0.01, 0.05]],
    minor=False,
)
ax[3, 0].set_yticks(
    [1.0, 0.0],
    labels=[freqFlabels[fF] for fF in [0.01, 0.05]],
    minor=False,
)

ax[0, 0].set_ylim(0.4, 1.2)
ax[0, 1].set_ylim(0.4, 1.2)
ax[0, 2].set_ylim(0.4, 1.2)
ax[2, 0].set_ylim(-0.5, 1.2)
ax[2, 1].set_ylim(-0.5, 1.2)
ax[2, 2].set_ylim(-0.5, 1.2)
ax[3, 0].set_ylim(-0.5, 1.2)
ax[3, 1].set_ylim(-0.5, 1.2)
ax[3, 2].set_ylim(-0.5, 1.2)

ax[0, 0].spines["bottom"].set_visible(False)
ax[0, 1].spines["bottom"].set_visible(False)
ax[0, 2].spines["bottom"].set_visible(False)
ax[1, 0].spines["top"].set_visible(False)
ax[1, 1].spines["top"].set_visible(False)
ax[1, 2].spines["top"].set_visible(False)

ax[0, 0].xaxis.tick_top()
ax[0, 1].xaxis.tick_top()
ax[0, 2].xaxis.tick_top()

ax[1, 0].set_ylabel("Signal-to-Noise Ratio", labelpad=10)
ax[2, 0].set_ylabel("Gain\nFluctuations\n(Noiseless)", labelpad=10, fontsize="small")
ax[3, 0].set_ylabel(
    "Throughput\nCalibration\n(Noiseless)", labelpad=10, fontsize="small"
)

ax[0, 1].set_yticks([])
ax[1, 1].set_yticks([])
ax[2, 1].set_yticks([])
ax[3, 1].set_yticks([])

ax[0, 2].sharey(ax[0, 1])
ax[1, 2].sharey(ax[1, 1])
ax[2, 2].sharey(ax[2, 1])
ax[3, 2].sharey(ax[3, 1])

ax[3, 0].set_xlabel(labels["amp"] + " " + r"[mK]")
ax[3, 1].set_xlabel(labels["width"] + " " + r"[MHz]")
ax[3, 2].set_xlabel(labels["numin"] + " " + r"[MHz]")

ax[0, 0].set_xlim(amin, amax)
ax[0, 1].set_xlim(wmin, wmax)
ax[0, 2].set_xlim(nmin, nmax)
# ax[1, 0].set_xlim(amin, amax)
# ax[1, 1].set_xlim(wmin, wmax)
# ax[1, 2].set_xlim(nmin, nmax)

print("setting limits", limits)
ax[0, 0].axvspan(amin, limits[0][0], color="k", alpha=0.1)
ax[0, 0].axvspan(limits[0][1], amax, color="k", alpha=0.1)
ax[0, 1].axvspan(wmin, limits[1][0], color="k", alpha=0.1)
ax[0, 1].axvspan(limits[1][1], wmax, color="k", alpha=0.1)
ax[0, 2].axvspan(nmin, limits[2][0], color="k", alpha=0.1)
ax[0, 2].axvspan(limits[2][1], nmax, color="k", alpha=0.1)
ax[1, 0].axvspan(amin, limits[0][0], color="k", alpha=0.1)
ax[1, 0].axvspan(limits[0][1], amax, color="k", alpha=0.1)
ax[1, 1].axvspan(wmin, limits[1][0], color="k", alpha=0.1)
ax[1, 1].axvspan(limits[1][1], wmax, color="k", alpha=0.1)
ax[1, 2].axvspan(nmin, limits[2][0], color="k", alpha=0.1)
ax[1, 2].axvspan(limits[2][1], nmax, color="k", alpha=0.1)
ax[2, 0].axvspan(amin, limits[0][0], color="k", alpha=0.1)
ax[2, 0].axvspan(limits[0][1], amax, color="k", alpha=0.1)
ax[2, 1].axvspan(wmin, limits[1][0], color="k", alpha=0.1)
ax[2, 1].axvspan(limits[1][1], wmax, color="k", alpha=0.1)
ax[2, 2].axvspan(nmin, limits[2][0], color="k", alpha=0.1)
ax[2, 2].axvspan(limits[2][1], nmax, color="k", alpha=0.1)
ax[3, 0].axvspan(amin, limits[0][0], color="k", alpha=0.1)
ax[3, 0].axvspan(limits[0][1], amax, color="k", alpha=0.1)
ax[3, 1].axvspan(wmin, limits[1][0], color="k", alpha=0.1)
ax[3, 1].axvspan(limits[1][1], wmax, color="k", alpha=0.1)
ax[3, 2].axvspan(nmin, limits[2][0], color="k", alpha=0.1)
ax[3, 2].axvspan(limits[2][1], nmax, color="k", alpha=0.1)

plt.plot([], [], **truthaxvlines, label="Truth")
plt.errorbar(
    [], [], xerr=[[], []], c=csColors[""], label=combineSigmalabels[""], lw=4.0
)
# plt.errorbar(
#     [],
#     [],
#     xerr=[[], []],
#     c="gray",
#     **achromaticerrbars,
#     markerfacecolor="none",
#     label="Achromatic",
# )
plt.errorbar(
    [], [], xerr=[[], []], c=csColors["4"], label=combineSigmalabels["4"], lw=4.0
)
# plt.errorbar(
#     [],
#     [],
#     xerr=[[], []],
#     c="gray",
#     **chromaticerrbars,
#     markerfacecolor="none",
#     label="Chromatic",
# )
plt.errorbar(
    [], [], xerr=[[], []], c=csColors["4 6"], label=combineSigmalabels["4 6"], lw=4.0
)
fig.legend(
    bbox_to_anchor=(0.5, 0.05), loc="center", borderaxespad=0, ncol=4, fontsize="small"
)

fig.suptitle(f"Cosmic Dawn Signal Constraints - Achromatic Beam")

print("saving")
plt.savefig(f"CD_table.pdf", dpi=300, bbox_inches="tight")

plt.show()
