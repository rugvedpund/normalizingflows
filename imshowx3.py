import numpy as np
import corner
import NormalizingFlow as nf
import matplotlib.pyplot as plt
import parser

##---------------------------------------------------------------------------##
# argparser

argparser = parser.create_parser()
args = argparser.parse_args()

# must set this
args.noisyT21 = True
args.diffCombineSigma = True

parser.prettyprint(args)

s, ll = nf.get_samplesAndLikelihood(args, plot="all")
npoints = [70, 71, 69]  # THIS IS WEIRD
print("Total number of samples: ", s.shape)
print("Max likelihood:", max(ll))
likelihoods = np.exp((ll - max(ll))).reshape(*npoints)

##---------------------------------------------------------------------------##
# setup corner plot params

cosmicdawn = True if args.fgFITS == "gsm16.fits" else False
fg = "CD" if cosmicdawn else "DA"
chromatic = "chromatic" if args.chromatic else "achromatic"
labels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
axlabels = {
    "a": "" r"$A$",
    "w": r"$\nu_{\rm rms}$",
    "n": r"$\nu_{\rm min}$",
}
units = {
    "a": "[mK]",
    "w": "[MHz]",
    "n": "[MHz]",
}


truths = [130, 20, 67.5] if cosmicdawn else [40, 14, 16.4]
s[:, 0] *= truths[0]

limits = dict()
mins = np.min(s, axis=0)
maxs = np.max(s, axis=0)
limits["amin"], limits["wmin"], limits["nmin"] = mins
limits["amax"], limits["wmax"], limits["nmax"] = maxs
limits["a"] = np.linspace(mins[0], maxs[0], npoints[1])
limits["w"] = np.linspace(mins[1], maxs[1], npoints[0])
limits["n"] = np.linspace(mins[2], maxs[2], npoints[2])
limits["da"] = (limits["a"][1] - limits["a"][0]) / 2
limits["dw"] = (limits["w"][1] - limits["w"][0]) / 2
limits["dn"] = (limits["n"][1] - limits["n"][0]) / 2

extent = dict()
extent["WvA"] = [
    limits["amin"] - limits["da"],
    limits["amax"] + limits["da"],
    limits["wmin"] - limits["dw"],
    limits["wmax"] + limits["dw"],
]
extent["NvA"] = [
    limits["amin"] - limits["da"],
    limits["amax"] + limits["da"],
    limits["nmin"] - limits["dn"],
    limits["nmax"] + limits["dn"],
]
extent["NvW"] = [
    limits["wmin"] - limits["dw"],
    limits["wmax"] + limits["dw"],
    limits["nmin"] - limits["dn"],
    limits["nmax"] + limits["dn"],
]

##---------------------------------------------------------------------------##
# helper functions


def norm(x):
    return x / np.max(x)


def quantile(x, likelihoods):
    q16, q50, q84 = corner.core.quantile(x, [0.16, 0.5, 0.84], weights=likelihoods)
    qm, qp = q50 - q16, q84 - q50
    return rf"${q50:.2f}^{{+{qp:.2f}}}_{{-{qm:.2f}}}$"


##---------------------------------------------------------------------------##
# plot samples and likelihoods


fig, ax = plt.subplots(3, 3, figsize=(8, 8), sharex="col")
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05)

ax[0, 0].step(limits["a"], likelihoods.sum(axis=(0, 2)), c="gray")
ax[1, 1].step(limits["w"], likelihoods.sum(axis=(1, 2)), c="gray")
ax[2, 2].step(limits["n"], likelihoods.sum(axis=(0, 1)), c="gray")

# THESE TRANSPOSES ARE WEIRD BUT CORRECT
ax[1, 0].imshow(
    norm(likelihoods.sum(axis=2)),
    extent=extent["WvA"],
    origin="lower",
    aspect="auto",
    cmap="Greys",
)
ax[2, 0].imshow(
    norm(likelihoods.sum(axis=0)).T,
    extent=extent["NvA"],
    origin="lower",
    aspect="auto",
    cmap="Greys",
)
im = ax[2, 1].imshow(
    norm(likelihoods.sum(axis=1)).T,
    extent=extent["NvW"],
    origin="lower",
    aspect="auto",
    cmap="Greys",
)

plt.colorbar(im, ax=ax[0, 2], label="Normalized Likelihood", orientation="vertical")
ax[0, 2].errorbar([], [], fmt="ro",xerr=[],yerr=[], label="Truth")
ax[0, 2].legend(loc="center")

ax[0, 0].set_title(
    rf"{axlabels['a']} = {quantile(limits['a'], likelihoods.sum(axis=(0, 2)))} {units['a']}"
)
ax[1, 1].set_title(
    rf"{axlabels['w']} = {quantile(limits['w'], likelihoods.sum(axis=(1, 2)))} {units['w']}"
)
ax[2, 2].set_title(
    rf"{axlabels['n']} = {quantile(limits['n'], likelihoods.sum(axis=(0, 1)))} {units['n']}"
)

ax[2, 0].set_xlabel(axlabels["a"] + " " + units["a"])
ax[2, 1].set_xlabel(axlabels["w"] + " " + units["w"])
ax[2, 2].set_xlabel(axlabels["n"] + " " + units["n"])
ax[1, 0].set_ylabel(axlabels["w"] + " " + units["w"])
ax[2, 0].set_ylabel(axlabels["n"] + " " + units["n"])

ax[0, 0].set_yticks([])
ax[1, 1].set_yticks([])
ax[2, 2].set_yticks([])

ax[2, 1].set_yticklabels([])

[l.set_rotation(45) for l in ax[2, 0].get_xticklabels()]
[l.set_rotation(45) for l in ax[2, 1].get_xticklabels()]
[l.set_rotation(45) for l in ax[2, 2].get_xticklabels()]
[l.set_rotation(45) for l in ax[1, 0].get_yticklabels()]
[l.set_rotation(45) for l in ax[2, 0].get_yticklabels()]

ax[1, 0].axhline(truths[1], c="r", alpha=0.5, lw=0.4)
ax[2, 0].axhline(truths[2], c="r", alpha=0.5, lw=0.4)
ax[2, 1].axhline(truths[2], c="r", alpha=0.5, lw=0.4)

ax[0, 0].axvline(truths[0], c="r", alpha=0.5, lw=0.4)
ax[1, 0].axvline(truths[0], c="r", alpha=0.5, lw=0.4)
ax[1, 1].axvline(truths[1], c="r", alpha=0.5, lw=0.4)
ax[2, 0].axvline(truths[0], c="r", alpha=0.5, lw=0.4)
ax[2, 1].axvline(truths[1], c="r", alpha=0.5, lw=0.4)
ax[2, 2].axvline(truths[2], c="r", alpha=0.5, lw=0.4)

ax[1, 0].plot(truths[0], truths[1], "ro")
ax[2, 0].plot(truths[0], truths[2], "ro")
ax[2, 1].plot(truths[1], truths[2], "ro")

ax[0, 1].set_visible(False)
ax[0, 2].axis("off")
ax[1, 2].set_visible(False)

plt.figtext(
    0.8,
    0.7,
    parser.argstostring(args),
    bbox=dict(facecolor="gray", alpha=0.5),
    fontsize="xx-small",
    ma="center",
)

plt.suptitle(f"{fg} for {chromatic} {labels[args.combineSigma]}, SNR={args.SNRpp:.0e}")
plt.show()
