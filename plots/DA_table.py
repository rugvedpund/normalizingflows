# create a pretty plot to replace the table

import NormalizingFlow as nf
import numpy as np
import matplotlib.pyplot as plt
import corner

args = nf.Args()
args.fgFITS, args.freqs = "ulsa.fits", "1 51"

truths = [40.0, 14.0, 16.4]
limits = [(0.01 * truths[0], 10 * truths[0]), (10, 30), (10, 30)]  # cube
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
achromaticerrbars = {"fmt": ".", "markersize": 9, "capsize": 4}
chromaticerrbars = {
    "fmt": "d",
    "markersize": 5,
    "capsize": 4,
    "xuplims": True,
    "markerfacecolor": "none",
}
snrpps = [1e8, 1e9, 1e10, 1e11]
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

fig, ax = plt.subplots(
    4,
    3,
    figsize=(10, 6),
    sharex="col",
    gridspec_kw=dict(height_ratios=[0.5, 1, 0.5, 0.5]),
)
fig.subplots_adjust(
    left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.05, hspace=0.1
)

for ins, nseed in enumerate([0]):
    args.noiseSeed = nseed
    print("Noise Seed", nseed)

    # for noiseless
    print("Noiseless")
    args.SNRpp = 1e24
    for ics, cs in enumerate(["", "4", "4 6"]):
        # achromatic
        args.combineSigma = cs
        args.appendLik = "_smallcube"
        args.chromatic = False
        try:
            s, ll = nf.get_samplesAndLikelihood(args, plot="all")
            assert ll.size == 342930
        except FileNotFoundError:
            s, ll = np.zeros_like(s), np.zeros_like(ll)
            print(f"SNRpp 1e24 achromatic {cs} {args.appendLik} not found")
            pass
        except AssertionError:
            print(ll.size)
            print(f"SNRpp 1e24 achromatic {cs} {args.appendLik} bad shape")
            pass
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]

        # plot
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[0, i].axvline(truths[i], **truthaxvlines)
            ax[0, i].errorbar(
                constraints[p],
                1 - 0.1 * ics - 0.02 * ins,
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                **achromaticerrbars,
                c=csColors[cs],
            )

        # # chromatic
        # args.chromatic = True
        # args.appendLik = "_1D"
        # try:
        #     s, ll = nf.get_samplesAndLikelihood(args, plot="A")
        # except FileNotFoundError:
        #     s,ll=np.zeros_like(s),np.zeros_like(ll)
        #     print(f"SNRpp 1e24 chromatic {cs} {args.appendLik} not found")
        #     pass
        # q16, q50, q84, q95 = corner.core.quantile(
        #     s[:, 0]*truths[0], [0.16, 0.5, 0.84, 0.95], weights=nf.exp(ll)
        # )
        # qm, qp = q50 - q16, q84 - q50
        # ax[0, 0].errorbar(
        #     # q50,
        #     q95,
        #     0.75 - 0.05 * ics,
        #     # xerr=[[qm], [qp]],
        #     xerr=[[q95], [0.0]],
        #     **chromaticerrbars,
        #     c="C" + str(ics),
        # )

    print("Noisy")
    # for noisy
    for isnrpp, snrpp in enumerate(snrpps):
        for ics, cs in enumerate(["", "4", "4 6"]):
            # achromatic
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
            constraints = nf.get_constraints(s, ll)
            constraints["amp"] *= truths[0]
            constraints["amp+"] *= truths[0]
            constraints["amp-"] *= truths[0]

            # plot
            for i, p in enumerate(["amp", "width", "numin"]):
                ax[1, i].axvline(truths[i], **truthaxvlines)
                ax[1, i].errorbar(
                    constraints[p],
                    isnrpp + 1 - 0.1 * ics - 0.02 * ins,
                    xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                    **achromaticerrbars,
                    c=csColors[cs],
                )

            # # chromatic
            # args.chromatic = True
            # args.appendLik = "_1D"
            # try:
            #     s, ll = nf.get_samplesAndLikelihood(args, plot="A")
            # except FileNotFoundError:
            #     s,ll=np.zeros_like(s),np.zeros_like(ll)
            #     print(f"SNRpp {snrpp:.0e} chromatic {cs} {args.appendLik} not found")
            #     pass
            # q16, q50, q84, q95 = corner.core.quantile(
            #     s[:, 0]*truths[0], [0.16, 0.5, 0.84, 0.95], weights=nf.exp(ll)
            # )
            # qm, qp = q50 - q16, q84 - q50
            # ax[1, 0].errorbar(
            #     # q50,
            #     q95,
            #     isnrpp + 0.75 - 0.05 * ics,
            #     # xerr=[[qm], [qp]],
            #     xerr=[[q95], [0.0]],
            #     **chromaticerrbars,
            #     c="C" + str(ics),
            # )

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
            constraints = nf.get_constraints(s, ll)
            constraints["amp"] *= truths[0]
            constraints["amp+"] *= truths[0]
            constraints["amp-"] *= truths[0]
            for i, p in enumerate(["amp", "width", "numin"]):
                ax[2, i].axvline(truths[i], **truthaxvlines)
                ax[2, i].errorbar(
                    constraints[p],
                    1 - igF - 0.1 * ics - 0.02 * ins,
                    xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                    **achromaticerrbars,
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
            constraints = nf.get_constraints(s, ll)
            constraints["amp"] *= truths[0]
            constraints["amp+"] *= truths[0]
            constraints["amp-"] *= truths[0]
            for i, p in enumerate(["amp", "width", "numin"]):
                ax[3, i].axvline(truths[i], **truthaxvlines)
                ax[3, i].errorbar(
                    constraints[p],
                    1 - ifF - 0.1 * ics - 0.02 * ins,
                    xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                    **achromaticerrbars,
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

ax[0, 0].set_ylim(0.5, 1.2)
ax[0, 1].set_ylim(0.5, 1.2)
ax[0, 2].set_ylim(0.5, 1.2)
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

amin, amax = -40.0, 180.0
wmin, wmax = 7.0, 23.0
nmin, nmax = 7.0, 23.0
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

plt.errorbar([], [], xerr=[[], []], c=csColors[""], label=combineSigmalabels[""], lw=4.0)
plt.errorbar([], [], xerr=[[], []], c=csColors["4"], label=combineSigmalabels["4"], lw=4.0)
# plt.errorbar([], [], xerr=[[], []], c="gray", **chromaticerrbars, label="Chromatic")
plt.errorbar([], [], xerr=[[], []], c=csColors["4 6"], label=combineSigmalabels["4 6"], lw=4.0)
plt.plot([], [], **truthaxvlines, label="Truth")
# plt.errorbar([], [], xerr=[[], []], c="gray", **achromaticerrbars, label="Achromatic")
fig.legend(
    bbox_to_anchor=(0.5, 0.05), loc="center", borderaxespad=0, ncol=4, fontsize="small"
)

fig.suptitle(f"Dark Ages Signal Constraints")

print("saving")
plt.savefig(f"DA_table.pdf", dpi=300, bbox_inches="tight")

plt.show()
