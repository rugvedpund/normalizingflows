import NormalizingFlow as nf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import corner
import makedf

root = os.path.abspath("/home/rugved/Files/LuSEE/normalizingflows/")

args = nf.Args()
args.appendLik = "_cmb"

# args.fgFITS, args.freqs = "ulsa.fits", "1 51"
args.fgFITS, args.freqs= "gsm16.fits", "51 101"
# args.chromatic = True

##---------------------------------------------------------------------------##
# plot params

cosmicdawn = True if args.fgFITS == "gsm16.fits" else False
colors = {"": "gray", "4": "C2", "4 6": "C3"}
lstyles = ["-.", ":", "-"]
labels = {
    "": r"$2^\circ$",
    "4": r"$2^\circ+4^\circ$",
    "4 6": r"$2^\circ+4^\circ+6^\circ$",
}
fg = "Cosmic Dawn" if cosmicdawn else "Dark Ages"
cc = "Chromatic" if args.chromatic else "Achromatic"
# snrpps = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
snrpps = [1e2,1e3,1e4] if cosmicdawn else [1e5, 1e6, 1e7]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
combineSigmas = ["", "4 6"]
truth = 2.725
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

pltdata = makedf.get_df(args, snrpps, seeds, combineSigmas)

for snrpp in snrpps:
    for ics, cs in enumerate(combineSigmas):
        plt.plot(
            truth * pltdata["s"][snrpp, cs],
            pltdata["med"][snrpp, cs],
            c=colors[cs],
            ls=lstyles[snrpps.index(snrpp)],
        )
        plt.fill_between(
            truth * pltdata["s"][snrpp, cs],
            pltdata["medlow"][snrpp, cs],
            pltdata["medhigh"][snrpp, cs],
            color=colors[cs],
            alpha=0.35,
        )
for snrpp in snrpps:
    plt.plot(
        [],
        [],
        c="k",
        label=rf"SNR={snrpp:.0e}",
        ls=lstyles[snrpps.index(snrpp)],
        lw=1.5,
    )
for cs in combineSigmas:
    plt.plot([], [], c=colors[cs], label=labels[cs], lw=4.0)

plt.axvline(truth, c="k", ls="--", label="Truth", lw=0.5)
plt.xlim((0.8,4.5) if cosmicdawn else (0.3, 10))
plt.ylim(0,1.1)
# plt.xscale("log")
plt.xlabel(r"$T_{\rm CMB} [K]$")
plt.title(f"CMB Monopole Likelihood for {fg} Model")
plt.legend()

fname = os.path.join(root,f"plots/{fg}_CMB.pdf")
print("saving to ", fname)
plt.savefig(fname, dpi=300, bbox_inches=None)

plt.show()

