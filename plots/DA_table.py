# create a pretty plot to replace the table

import NormalizingFlow as nf
import numpy as np
import matplotlib.pyplot as plt
import corner

args = nf.Args()
args.fgFITS, args.freqs = "ulsa.fits", "1 51"
# args.fgFITS, args.freqs = "gsm16.fits", "51 101"
args.appendLik = "_cube"
args.noiseSeed = 2
args.chromatic = False

truths = [40, 14, 16.4] if args.fgFITS == "ulsa.fits" else [130, 20, 67.5]
labels = {"amp": r"$A$", "width": r"$\nu_{\rm rms}$", "numin": r"$\nu_{\rm min}$"}
fg = "Dark Ages" if args.fgFITS == "ulsa.fits" else "Cosmic Dawn"
chromatic = "Chromatic" if args.chromatic == "True" else "Achromatic"
sigmaLabels = {2: "$2^\circ$", 4: "$4^\circ$", 6: "$6^\circ$"}
combineSigmalabels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
ylims = (2e7, 5e14) if args.fgFITS == "ulsa.fits" else (2e5, 5e14)

snrpps = [1e8, 1e9, 1e10, 1e11]
# snrpps = [1e5, 1e6, 1e7, 1e8]
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

fig, ax = plt.subplots(1, 3, figsize=(10, 8))

# for noisy
for isnrpp, snrpp in enumerate(snrpps):
    for ics, cs in enumerate(["", "4", "4 6"]):
        args.SNRpp = snrpp
        args.combineSigma = cs
        try:
            s, ll = nf.get_samplesAndLikelihood(args, plot="all")
        except FileNotFoundError:
            print(f"Achromatic {snrpp:.0e} {cs} not found, continuing")
            continue
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[i].errorbar(
                constraints[p],
                isnrpp + 1 -0.05*ics ,
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                fmt=".",
                capsize=4,
                c="C" + str(ics),
                label=combineSigmalabels[cs] if isnrpp == 0 else None,
            )

# for noiseless
args.SNRpp = 1e24
s, ll = nf.get_samplesAndLikelihood(args, plot="all")
constraints = nf.get_constraints(s, ll)
constraints["amp"] *= truths[0]
constraints["amp+"] *= truths[0]
constraints["amp-"] *= truths[0]
for i, p in enumerate(["amp", "width", "numin"]):
    for ics, cs in enumerate(["", "4", "4 6"]):
        # ax[i].axhline(snrpps[-1]*1e2,c='k',alpha=0.1)
        ax[i].errorbar(
            constraints[p],
            len(snrpps) + 1-0.05*ics,
            xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
            fmt=".",
            capsize=4,
            c="C" + str(ics),
        )
        ax[i].axvline(truths[i], c="gray", ls="--", lw=0.7)
        ax[i].set_xlabel(labels[p])

##---------------------------------------------------------------------------##

# for chromatic
args.chromatic = True
for isnrpp, snrpp in enumerate(snrpps):
    for ics, cs in enumerate(["", "4", "4 6"]):
        args.SNRpp = snrpp
        args.combineSigma = cs
        try:
            s, ll = nf.get_samplesAndLikelihood(args, plot="all")
        except FileNotFoundError:
            print(f"chromatic {snrpp:.0e} {cs} not found, continuing")
            continue
        constraints = nf.get_constraints(s, ll)
        constraints["amp"] *= truths[0]
        constraints["amp+"] *= truths[0]
        constraints["amp-"] *= truths[0]
        for i, p in enumerate(["amp", "width", "numin"]):
            ax[i].errorbar(
                constraints[p],
                isnrpp + 0.8-0.05*ics, 
                xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
                fmt="d",
                markersize=4,
                capsize=4,
                c="C" + str(ics),
                label=combineSigmalabels[cs] if isnrpp == 0 else None,
            )

# for noiseless
args.SNRpp = 1e24
s, ll = nf.get_samplesAndLikelihood(args, plot="all")
constraints = nf.get_constraints(s, ll)
constraints["amp"] *= truths[0]
constraints["amp+"] *= truths[0]
constraints["amp-"] *= truths[0]
for i, p in enumerate(["amp", "width", "numin"]):
    for ics, cs in enumerate(["", "4", "4 6"]):
        # ax[i].axhline(snrpps[-1]*1e2,c='k',alpha=0.1)
        ax[i].errorbar(
            constraints[p],
            len(snrpps) + 0.8 - 0.05*ics,
            xerr=[[constraints[p + "-"]], [constraints[p + "+"]]],
            fmt="d",
            markersize=4,
            capsize=4,
            c="C" + str(ics),
        )

ax[0].set_yticks(
    np.arange(1, len(snrpps) + 2),
    labels=[snrpplabels[snrpp] for snrpp in snrpps] + ["Zero"],
    minor=False,
)
# for i in range(3): ax[i].set_ylim(ylims)
ax[0].set_ylabel(r"SNR$_{\rm pp}$")
ax[1].get_yaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
plt.plot([], [], c="k", ls="--", lw=0.7, label="Truth")
fig.suptitle(f"{fg} Signal Constraints")

plt.legend()

plt.tight_layout()
plt.show()
