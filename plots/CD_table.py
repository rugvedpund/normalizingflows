# create a pretty plot to replace the table

import NormalizingFlow as nf
import numpy as np
import matplotlib.pyplot as plt
import corner

args = nf.Args()
args.fgFITS, args.freqs = "gsm16.fits", "51 101"
args.noiseSeed = 0

truths = [130, 20, 67.5]
limits = [(0.01 * truths[0], 10 * truths[0]), (10, 40), (50, 90)]  # cube
labels = {"amp": r"$A$", "width": r"$\nu_{\rm rms}$", "numin": r"$\nu_{\rm min}$"}
chromatic = "Chromatic" if args.chromatic == "True" else "Achromatic"
sigmaLabels = {2: "$2^\circ$", 4: "$4^\circ$", 6: "$6^\circ$"}
combineSigmalabels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
truthaxvlines = {"c": "k", "ls": ":", "lw": 0.7}
achromaticerrbars = {"fmt": ".", "markersize": 9, "capsize": 4}
chromaticerrbars = {"fmt": "d", "markersize": 5, "capsize": 4}
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

##---------------------------------------------------------------------------##

fig, ax = plt.subplots(
    2,
    3,
    figsize=(10, 8),
    sharex="col",
    gridspec_kw=dict(height_ratios=[0.5, 2]),
)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.9, wspace=0.05, hspace=0.1)

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
        continue
    except AssertionError:
        print(ll.size)
        print(f"SNRpp 1e24 achromatic {cs} {args.appendLik} bad shape")
        continue
    constraints = nf.get_constraints(s, ll)
    constraints["amp"] *= truths[0]
    constraints["amp+"] *= truths[0]
    constraints["amp-"] *= truths[0]

    # plot
    for i, p in enumerate(["amp", "width", "numin"]):
        ax[0, i].axvline(truths[i], **truthaxvlines)
        ax[0, i].errorbar(
            constraints[p],
            1 - 0.05 * ics,
            xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
            **achromaticerrbars,
            c="C" + str(ics),
        )

    # chromatic
    args.chromatic = True
    args.appendLik = "_cube"
    try:
        s, ll = nf.get_samplesAndLikelihood(args, plot="all")
        assert ll.size == 342930
    except FileNotFoundError:
        print(f"SNRpp 1e24 chromatic {cs} {args.appendLik} not found")
        continue
    except AssertionError:
        print(ll.size)
        print(f"SNRpp 1e24 chromatic {cs} {args.appendLik} bad shape")
        continue
    constraints = nf.get_constraints(s, ll)
    constraints["amp"] *= truths[0]
    constraints["amp+"] *= truths[0]
    constraints["amp-"] *= truths[0]
    for i, p in enumerate(["amp", "width", "numin"]):
        ax[0, i].errorbar(
            constraints[p],
            0.75 - 0.05 * ics,
            xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
            **chromaticerrbars,
            c="C" + str(ics),
        )


print("Noisy")
# for noisy
for isnrpp, snrpp in enumerate(snrpps):
    for ics, cs in enumerate(["", "4", "4 6"]):
        #achromatic
        args.appendLik = "_cube"
        args.chromatic = False
        args.SNRpp = snrpp
        args.combineSigma = cs
        try:
            s, ll = nf.get_samplesAndLikelihood(args, plot="all")
            assert ll.size == 342930
        except FileNotFoundError:
            print(f"SNRpp {snrpp:.0e} achromatic {cs} {args.appendLik} not found")
            continue
        except AssertionError:
            print(ll.size)
            print(f"SNRpp {snrpp:.0e} achromatic {cs} {args.appendLik} bad shape")
            continue
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]

        #plot
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[1, i].axvline(truths[i], **truthaxvlines)
            ax[1, i].errorbar(
                constraints[p],
                isnrpp + 1 - 0.05 * ics,
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                **achromaticerrbars,
                c="C" + str(ics),
            )
        args.chromatic = True
        try:
            s, ll = nf.get_samplesAndLikelihood(args, plot="all")
            assert ll.size == 342930
        except FileNotFoundError:
            print( f"SNRpp {snrpp:.0e} chromatic {cs} {args.appendLik} not found")
            continue
        except AssertionError:
            print(ll.size)
            print( f"SNRpp {snrpp:.0e} achromatic {cs} {args.appendLik} bad shape")
            continue
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[1, i].errorbar(
                constraints[p],
                isnrpp + 0.75 - 0.05 * ics,
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                **chromaticerrbars,
                c="C" + str(ics),
            )

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

ax[0, 0].set_ylim(0.5, 1.2)
ax[0, 1].set_ylim(0.5, 1.2)
ax[0, 2].set_ylim(0.5, 1.2)

ax[0, 0].spines["bottom"].set_visible(False)
ax[0, 1].spines["bottom"].set_visible(False)
ax[0, 2].spines["bottom"].set_visible(False)
ax[1, 0].spines["top"].set_visible(False)
ax[1, 1].spines["top"].set_visible(False)
ax[1, 2].spines["top"].set_visible(False)

ax[0, 0].xaxis.tick_top()
ax[0, 1].xaxis.tick_top()
ax[0, 2].xaxis.tick_top()
ax[0, 1].set_yticks([])
ax[0, 2].sharey(ax[0, 1])

# ax[0, 0].set_ylabel("Noiseless", labelpad=10, rotation=0)
ax[1, 0].set_ylabel("Signal-to-Noise Ratio", labelpad=10)

ax[1, 1].set_yticks([])
ax[1, 2].sharey(ax[1, 1])

ax[1, 0].set_xlabel(labels["amp"] + " " + r"[mK]")
ax[1, 1].set_xlabel(labels["width"] + " " + r"[MHz]")
ax[1, 2].set_xlabel(labels["numin"] + " " + r"[MHz]")

amin, amax = -100.0, 1500.0
wmin, wmax = 5.0, 50.0
nmin, nmax = 45.0, 95.0
ax[0, 0].set_xlim(amin, amax)
ax[0, 1].set_xlim(wmin, wmax)
ax[0, 2].set_xlim(nmin, nmax)
ax[1, 0].set_xlim(amin, amax)
ax[1, 1].set_xlim(wmin, wmax)
ax[1, 2].set_xlim(nmin, nmax)

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

plt.plot([], [], **truthaxvlines, label="Truth")
plt.errorbar([], [], xerr=[[], []], c="gray", **achromaticerrbars, label="Achromatic")
plt.errorbar([], [], xerr=[[], []], c="gray", **chromaticerrbars, label="Chromatic")
plt.errorbar([], [], xerr=[[], []], c="C0", label=combineSigmalabels[""], lw=4.0)
plt.errorbar([], [], xerr=[[], []], c="C1", label=combineSigmalabels["4"], lw=4.0)
plt.errorbar([], [], xerr=[[], []], c="C2", label=combineSigmalabels["4 6"], lw=4.0)
fig.legend(bbox_to_anchor=(0.9, 0.5), loc="center", borderaxespad=0)

fig.suptitle(f"Cosmic Dawn Signal Constraints")

print("saving")
plt.savefig(f"CD_table.pdf", dpi=300, bbox_inches="tight")

plt.show()