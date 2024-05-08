import matplotlib.pyplot as plt
import numpy as np
import NormalizingFlow as nf

##---------------------------------------------------------------------------##

tseeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
noiseSeeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
truths = [40, 130]


def format_constraints(truth,nseedmeans,tseedmeans):
    sysbias = np.mean(tseedmeans) - truth
    syserror = np.std(tseedmeans)
    statbias = np.mean(nseedmeans) - truth
    staterror = np.std(nseedmeans)
    return rf"${truth}\,$mK (truth) ${sysbias:+.1f}^{{+{syserror:.1f}}}_{{-{syserror:.1f}}}$(sinf)${statbias:+.1f}^{{+{staterror:.1f}}}_{{-{staterror:.1f}}}$(bias)"

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.subplots_adjust(top=0.8, bottom=0.2, wspace=0.05)

##---------------------------------------------------------------------------##
# dark ages

args = nf.Args()
args.fgFITS, args.freqs = "ulsa.fits", "1 51"
args.SNRpp = 1e9

args.SNRpp = 1e10
args.chromatic = True
args.combineSigma = "4 6"

nseedmeans = []
for nseed in noiseSeeds:
    args.appendLik = "_1D"
    args.noiseSeed = nseed
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= truths[0]
    args.appendLik = ""
    args.noiseSeed = 0
    constraints = nf.get_constraints(s, ll)
    nseedmeans.append(constraints["amp"])
    ax[0].plot(s[:, 0], nf.exp(ll), alpha=0.4, c=f"C1")

tseedmeans = []
for tseed in tseeds:
    args.append = f"_tS{tseed}"
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= truths[0]
    constraints = nf.get_constraints(s, ll)
    tseedmeans.append(constraints["amp"])
    args.append = "_SVD"
    ax[0].plot(s[:, 0], nf.exp(ll), alpha=0.4, c=f"C0")

# ref
fullerror = format_constraints(truths[0],nseedmeans,tseedmeans)
ax[0].set_title(f"Dark Ages\n {fullerror}", fontsize="medium")

##---------------------------------------------------------------------------##
# cosmic dawn

args = nf.Args()
args.fgFITS, args.freqs = "gsm16.fits", "51 101"
args.SNRpp = 1e6

args.SNRpp = 1e7
args.chromatic = True
args.combineSigma = "4 6"

nseedmeans = []
for nseed in noiseSeeds:
    args.appendLik = "_1D"
    args.noiseSeed = nseed
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= truths[1]
    args.appendLik = ""
    args.noiseSeed = 0
    constraints = nf.get_constraints(s, ll)
    nseedmeans.append(constraints["amp"])
    ax[1].plot(s[:, 0], nf.exp(ll), alpha=0.4, c=f"C1")

tseedmeans = []
for tseed in tseeds:
    args.append = f"_tS{tseed}"
    s, ll = nf.get_samplesAndLikelihood(args, plot="A")
    s[:, 0] *= truths[1]
    constraints = nf.get_constraints(s, ll)
    tseedmeans.append(constraints["amp"])
    args.append = "_SVD"
    ax[1].plot(s[:, 0], nf.exp(ll), alpha=0.4, c=f"C0")

# ref
fullerror = format_constraints(truths[1], nseedmeans, tseedmeans)
ax[1].set_title(f"Cosmic Dawn\n {fullerror}", fontsize="medium")

##---------------------------------------------------------------------------##

ax[0].set_xlabel(r"$A$ [mK]")
ax[0].set_ylabel("Likelihood")
ax[0].set_ylim(0, 1.2)
ax[0].axvline(truths[0], c="gray", ls="--")

# ax[0].set_xlim(10, 70)
# ax[1].set_xlim(80, 180)

# # for 1e8,1e5
# ax[0].set_xlim(0.1, 200)
# ax[1].set_xlim(0.1, 400)

# # for 1e10,1e7
# ax[0].set_xlim(25, 55)
# ax[1].set_xlim(115, 145)

# for chromatic, 2+4+6
ax[0].set_xlim(10, 160)
ax[1].set_xlim(80, 180)

ax[1].set_xlabel(r"$A$ [mK]")
ax[1].set_ylim(0, 1.2)
ax[1].axvline(truths[1], c="gray", ls="--")

plt.plot([], [], "C0", label="Systematic Variation", lw=3.0, alpha=0.5)
plt.plot([], [], "C1", label="Noise Variation", lw=3.0, alpha=0.5)
plt.plot([], [], c="gray", ls="--", label="Truth", lw=1.5)
fig.legend(
    bbox_to_anchor=(0.5, 0.05), loc="center", borderaxespad=0, ncol=3, fontsize="small"
)

# fig.suptitle(
#     r"Likelihood Biases: SINF Systematics and Noise Realizations"+"\n"+r"for $2^\circ$ Achromatic Beam"
# )

fig.suptitle(
    r"Likelihood Biases: SINF Systematics and Noise Realizations"+"\n"+r"for $2^\circ+4^\circ+6^\circ$ Chromatic Beam"
)

print("Saving")
# plt.savefig("torchSeeds.pdf", bbox_inches="tight", dpi=300)
plt.savefig("torchSeeds2.pdf", bbox_inches="tight", dpi=300)

plt.show()
