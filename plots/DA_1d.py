import numpy as np
import NormalizingFlow as nf
import matplotlib.pyplot as plt
import pandas as pd

args = nf.Args()

args.fgFITS, args.freqs='gsm16.fits', '51 101'
# args.fgFITS, args.freqs = "ulsa.fits", "1 51"
# args.chromatic = False
args.chromatic = True

# snrpps = [1e4, 1e5, 1e6]
snrpps = [1e5, 1e6, 1e7]
# snrpps = [1e7, 1e8, 1e9]
# snrpps = [1e8, 1e9, 1e10]
# snrpps = [1e9, 1e10, 1e11]
# snrpps = [1e9, 1e12, 1e24]

args.appendLik = "_1D"
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
combineSigmas = ["", "4", "4 6"]
colors = {"": "dimgray", "4": "C2", "4 6": "C3"}
lstyles = [":", "-.", "-"]
labels = {
    "": r"$2^\circ$",
    "4": r"$2^\circ+4^\circ$",
    "4 6": r"$2^\circ+4^\circ+6^\circ$",
}
alphas = {"": 0.05, "4": 0.1, "4 6": 0.3}
chromatic = "Chromatic" if args.chromatic else "Achromatic"
fg = "Dark Ages" if args.fgFITS == "ulsa.fits" else "Cosmic Dawn"
afactor = 40 if args.fgFITS == "ulsa.fits" else 130
snrpplabels = {
    1e4: r"SNR=$10^4$",
    1e5: r"SNR=$10^5$",
    1e6: r"SNR=$10^6$",
    1e7: r"SNR=$10^7$",
    1e8: r"SNR=$10^8$",
    1e9: r"SNR=$10^9$",
    1e10: r"SNR=$10^{10}$",
    1e11: r"SNR=$10^{11}$",
    1e12: r"SNR=$10^{12}$",
    1e13: r"$10^{13}$",
    1e24: "Noiseless",
}

rawdata = list()
for snrpp in snrpps:
    for seed in seeds:
        for cs in combineSigmas:
            args.SNRpp = snrpp
            args.noiseSeed = seed
            args.combineSigma = cs
            s, ll = nf.get_samplesAndLikelihood(args, plot="A")
            rawdata.append(
                {"snrpp": snrpp, "seed": seed, "cs": cs, "s": s[:, 0], "l": nf.exp(ll)}
            )
df = pd.DataFrame(rawdata).set_index(["snrpp", "seed", "cs"])
pltdata = pd.DataFrame(
    columns=["s", "med", "std", "medlow", "medhigh"],
    index=pd.MultiIndex.from_product([snrpps, combineSigmas], names=["snrpp", "cs"]),
)
# create pltdata with medians and stds
for snrpp in snrpps:
    for cs in combineSigmas:
        pltdata["s"][snrpp, cs] = np.median(
            [df["s"][snrpp, seed, cs] for seed in seeds], axis=0
        )
        pltdata["med"][snrpp, cs] = np.median(
            [df["l"][snrpp, seed, cs] for seed in seeds], axis=0
        )
        pltdata["std"][snrpp, cs] = np.std(
            [df["l"][snrpp, seed, cs] for seed in seeds], axis=0
        )
        pltdata["medlow"][snrpp, cs] = np.clip(
            pltdata["med"][snrpp, cs] - pltdata["std"][snrpp, cs], 0, None
        )
        pltdata["medhigh"][snrpp, cs] = np.clip(
            pltdata["med"][snrpp, cs] + pltdata["std"][snrpp, cs], 0, None
        )

# plot
for snrpp in snrpps:
    for cs in combineSigmas:
        plt.plot(
            afactor * pltdata["s"][snrpp, cs],
            pltdata["med"][snrpp, cs],
            c=colors[cs],
            ls=lstyles[snrpps.index(snrpp)],
        )
        plt.fill_between(
            afactor * pltdata["s"][snrpp, cs],
            pltdata["medlow"][snrpp, cs],
            pltdata["medhigh"][snrpp, cs],
            color=colors[cs],
            alpha=alphas[cs],
        )

# clean legend and axes
for snrpp in snrpps:
    plt.plot([], [], c="k", ls=lstyles[snrpps.index(snrpp)],label=rf"{snrpplabels[snrpp]}")
for cs in combineSigmas:
    plt.plot([], [], c=colors[cs], label=labels[cs], lw=4.0)
plt.axvline(afactor, color="k", ls="--", label="Truth")
plt.ylabel("Likelihood")
plt.xlabel("Amplitude [mK]")
plt.ylim(0, 1.15)
plt.xlim(0.01 * afactor, 1e4)
plt.xscale("log")
plt.title(f"{fg} Signal Amplitude Likelihood - {chromatic} Beam")
plt.legend()

fname = f"{fg}_{chromatic}_1dAmp.pdf"
print("saving to ", fname)
plt.savefig(fname, dpi=300, bbox_inches="tight")

plt.show()
