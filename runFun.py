import pandas as pd
import seaborn as sns
import numpy as np
import torch
import NormalizingFlow as nf
import lusee
import parser
import corner
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

##---------------------------------------------------------------------------##

argparser = parser.create_parser()
args = argparser.parse_args()
args.noisyT21 = True
args.diffCombineSigma = True
parser.prettyprint(args)

fname = nf.get_fname(args)

print(f"loading flow from {fname}")
flow = nf.FlowAnalyzerV2(nocuda=False, loadPath=fname)
flow.set_fg(args)

if args.fgFITS == "ulsa.fits":
    print("using DA model")
    t21 = lusee.MonoSkyModels.T_DarkAges_Scaled(
        flow.freqs, nu_rms=14, nu_min=16.4, A=0.04
    )
    cosmicdawn = False
elif args.fgFITS == "gsm16.fits":
    print("using CD model")
    t21 = lusee.MonoSkyModels.T_CosmicDawn_Scaled(
        flow.freqs, nu_rms=20, nu_min=67.5, A=0.130
    )
    cosmicdawn = True
flow.set_t21(t21)
if args.retrain:
    flow.train(
        flow.train_data, flow.validate_data, nocuda=False, savePath=fname, retrain=True
    )

if cosmicdawn:
    priorlow, priorhigh = [0.01, 10, 50], [10, 40, 90]
else:
    priorlow, priorhigh = [0.01, 10, 10], [10, 30, 30]


def like(x):
    return flow.get_likelihoodFromSamplesGAME(x, priorlow=priorlow, priorhigh=priorhigh)


##---------------------------------------------------------------------------##
# let's have some fun


def proj(fg):
    return flow.eve.T @ (fg - fg.mean(axis=1)[:, None])


cov = np.cov(flow.fgcut)
covss = np.cov(flow.fgss)
corr = np.corrcoef(flow.fgcut)
corrss = np.corrcoef(flow.fgss)
pcov = np.cov(proj(flow.fgcut))
pcovss = np.cov(proj(flow.fgss))
pcorr = np.corrcoef(proj(flow.fgcut))
pcorrss = np.corrcoef(proj(flow.fgss))

# plt.figure(figsize=(8, 8))
# sns.heatmap(corr, annot=True, cmap="viridis", square=True)
# plt.title("corr full")
# plt.show()

datasmall = proj(flow.fgcut).reshape(5,10,-1).mean(axis=1)
databig = (proj(flow.fgcut))/flow.rms[:, None]


# kwargs = dict(origin="lower", cmap="viridis")

# fig, ax = plt.subplots(4, 2, figsize=(4,8))

# im0 = ax[0, 0].imshow(corr, **kwargs)
# plt.colorbar(im0, ax=ax[0, 0], shrink=0.9)
# ax[0, 0].set_title("corr full", fontsize="small")

# im1 = ax[0, 1].imshow(corrss, **kwargs)
# plt.colorbar(im1, ax=ax[0, 1], shrink=0.9)
# ax[0, 1].set_title("corr subsample", fontsize="small")

# im6 = ax[2, 0].imshow(pcorr, **kwargs)
# plt.colorbar(im6, ax=ax[2, 0], shrink=0.9)
# ax[2, 0].set_title("proj corr full", fontsize="small")

# im7 = ax[2, 1].imshow(pcorrss, **kwargs)
# plt.colorbar(im7, ax=ax[2, 1], shrink=0.9)
# ax[2, 1].set_title("proj corr subsample", fontsize="small")

# kwargs = dict(origin="lower", cmap="viridis", norm=mcolors.SymLogNorm(linthresh=1e-5))

# im3 = ax[1, 0].imshow(cov, **kwargs)
# plt.colorbar(im3, ax=ax[1, 0], shrink=0.9)
# ax[1, 0].set_title("cov full", fontsize="small")

# im4 = ax[1, 1].imshow(covss, **kwargs)
# plt.colorbar(im4, ax=ax[1, 1], shrink=0.9)
# ax[1, 1].set_title("cov subsample", fontsize="small")

# im9 = ax[3, 0].imshow(pcov, **kwargs)
# plt.colorbar(im9, ax=ax[3, 0], shrink=0.9)
# ax[3, 0].set_title("proj cov full", fontsize="small")

# im10 = ax[3, 1].imshow(pcovss, **kwargs)
# plt.colorbar(im10, ax=ax[3, 1], shrink=0.9)
# ax[3, 1].set_title("proj cov subsample", fontsize="small")

# plt.tight_layout()
# plt.show()

import ipdb; ipdb.set_trace()
