# create a pretty plot to replace the table

import NormalizingFlow as nf
import numpy as np
import matplotlib.pyplot as plt
import corner

args = nf.Args()
args.fgFITS, args.freqs = "ulsa.fits", "1 51"
args.noiseSeed = 0

truths = [40.0, 14.0, 16.4]
limits = [(0.01 * truths[0], 10 * truths[0]), (10, 30), (10, 30)]  # cube
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
noises = [0.1, 0.025, 0.005, 0.0001, 0.00001]
noiselabels = {
        0.1: r"$100$ mK",
        0.025: r"$25$ mK",
        0.005: r"$5$ mK",
        0.0001: r"$0.1$ mK",
        0.00001: r"$0.01$ mK",
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
    figsize=(10, 8),
    sharex="col",
    gridspec_kw=dict(height_ratios=[0.5, 2, 0.5, 0.5]),
)
fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.05, hspace=0.1)

# for noiseless
print("Noiseless")
args.noise = 0.0
for ics, cs in enumerate(["", "4", "4 6"]):
    args.combineSigma = cs
    args.appendLik = "_cube"
    args.chromatic = False
    try:
        s, ll = nf.get_samplesAndLikelihood(args, plot="all")
        assert ll.size == 342930
    except FileNotFoundError:
        print(f"noise 0.0 achromatic {cs} {args.appendLik} not found")
        continue
    except AssertionError:
        print(ll.size)
        print(f"noise 0.0 achromatic {cs} {args.appendLik} bad shape")
        continue
    constraints = nf.get_constraints(s, ll)
    constraints["amp"] *= truths[0]
    constraints["amp+"] *= truths[0]
    constraints["amp-"] *= truths[0]
    for i, p in enumerate(["amp", "width", "numin"]):
        ax[0, i].axvline(truths[i], **truthaxvlines)
        ax[0, i].errorbar(
            constraints[p],
            1 - 0.05 * ics,
            xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
            **achromaticerrbars,
            c="C" + str(ics),
        )
    args.chromatic = True
    args.appendLik = "_cube"
    try:
        s, ll = nf.get_samplesAndLikelihood(args, plot="all")
        assert ll.size == 342930
    except FileNotFoundError:
        print(f"noise 0.0 chromatic {cs} {args.appendLik} not found")
        continue
    except AssertionError:
        print(ll.size)
        print(f"noise 0.0 achromatic {cs} {args.appendLik} bad shape")
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

##---------------------------------------------------------------------------##

print("Noisy")
# for noisy
for ni,n in enumerate(noises):
    for ics, cs in enumerate(["", "4", "4 6"]):
        args.appendLik = "_cube"
        args.chromatic = False
        args.noise = n
        args.combineSigma = cs
        try:
            s, ll = nf.get_samplesAndLikelihood(args, plot="all")
            assert ll.size == 342930
        except FileNotFoundError:
            print( f"noise {n} achromatic {cs} {args.appendLik} not found")
            continue
        except AssertionError:
            print(ll.size)
            print(f"noise {n} achromatic {cs} {args.appendLik} bad shape")
            continue
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[1, i].axvline(truths[i], **truthaxvlines)
            ax[1, i].errorbar(
                constraints[p],
                ni + 1 - 0.05 * ics,
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                **achromaticerrbars,
                c="C" + str(ics),
            )
        args.chromatic = True
        try:
            s, ll = nf.get_samplesAndLikelihood(args, plot="all")
            assert ll.size == 342930
        except FileNotFoundError:
            print( f"noise {n} chromatic {cs} {args.appendLik} not found")
            continue
        except AssertionError:
            print(ll.size)
            print(f"noise {n} achromatic {cs} {args.appendLik} bad shape")
            continue
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[1, i].errorbar(
                constraints[p],
                ni + 0.75 - 0.05 * ics,
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                **chromaticerrbars,
                c="C" + str(ics),
            )

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
            print(f"gainF achromatic {cs} {args.appendLik} not found")
            continue
        except AssertionError:
            print(ll.size)
            print(f"gainF achromatic {cs} {args.appendLik} bad shape")
            continue
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[2, i].axvline(truths[i], **truthaxvlines)
            ax[2, i].errorbar(
                constraints[p],
                1 - igF - 0.05 * ics,
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                **achromaticerrbars,
                c="C" + str(ics),
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
            print(f"freqF achromatic {cs} {args.appendLik} not found")
            continue
        except AssertionError:
            print(ll.size)
            print(f"freqF achromatic {cs} {args.appendLik} bad shape")
            continue
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[3, i].axvline(truths[i], **truthaxvlines)
            ax[3, i].errorbar(
                constraints[p],
                1 - ifF - 0.05 * ics,
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                **achromaticerrbars,
                c="C" + str(ics),
            )


##---------------------------------------------------------------------------##
print("done")

ax[0, 0].set_yticks(
    [1.0],
    labels=["Noiseless"],
    minor=False,
)
ax[1, 0].set_yticks(
    np.arange(1, len(noises) + 1),
    labels=[noiselabels[n] for n in noises],
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

ax[1, 0].set_ylabel("Noise Level")
ax[2, 0].set_ylabel("Gain\nFluctuations\n(Noiseless)", labelpad=10, fontsize="small")
ax[3, 0].set_ylabel("Throughput\nCalibration\n(Noiseless)", labelpad=10, fontsize="small")

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

amin, amax = -50.0, 500.0
wmin, wmax = 5.0, 35.0
nmin, nmax = 5.0, 35.0
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
plt.errorbar([], [], xerr=[[], []], c="C0", label=combineSigmalabels[""], lw=4.0)
plt.errorbar([], [], xerr=[[], []], c="gray", **achromaticerrbars, label="Achromatic")
plt.errorbar([], [], xerr=[[], []], c="C1", label=combineSigmalabels["4"], lw=4.0)
plt.errorbar([], [], xerr=[[], []], c="gray", **chromaticerrbars, label="Chromatic")
plt.errorbar([], [], xerr=[[], []], c="C2", label=combineSigmalabels["4 6"], lw=4.0)
fig.legend(bbox_to_anchor=(0.5, 0.05), loc="center", borderaxespad=0, ncol=3, fontsize="small")

fig.suptitle(f"Dark Ages Signal Constraints")

print("saving")
plt.savefig("DA_table_flatnoise.pdf", dpi=300, bbox_inches="tight")


plt.show()
