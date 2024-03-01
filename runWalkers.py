# Example:
# python runFast.py --noise 0.0 --combineSigma '4 6' --freqs '1 51' --fgFITS 'ulsa.fits'

import numpy as np
import torch
import fitsio
import NormalizingFlow as nf
import walkers
import lusee
import parser
import os
import corner
import matplotlib.pyplot as plt

##---------------------------------------------------------------------------##
# argparser block

argparser = parser.create_parser()
args = argparser.parse_args()

# must have --noisyT21 and --diffCombineSigma!!!
args.noisyT21 = True
args.diffCombineSigma = True

if args.appendLik == "":
    args.appendLik = "_walkers"

parser.prettyprint(args)

##---------------------------------------------------------------------------##

# set seed
print(f"setting noise seed {args.noiseSeed} and torch seed {args.torchSeed}")
np.random.seed(args.noiseSeed)
torch.manual_seed(args.torchSeed)
torch.cuda.manual_seed_all(args.torchSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
torch.set_default_tensor_type("torch.cuda.FloatTensor")

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
# %%

# samples=np.loadtxt('/home/rugved/Files/LuSEE/normalizingflows/tests/new_samplesandlls',unpack=True)[:-1].T
# breakpoint()
# s,loglikelihoods=flow.get_likelihoodFromSamples(samples,cmb=False)
# breakpoint()

npoints1 = 100
nwalkers = 100
nsteps = 10
nsteps2 = 100
walks = walkers.Walkers(args, nwalkers, nsteps)
walks.setInitialKWargs()
s, ll = walks.runInitial1DLikelihoods(flow, npoints1, cmb=False)
walkerparams = walks.extractWalkerStart(s, ll, npoints1)
wsteps = walks.walkWalkers(walkerparams)
s2, ll2 = walks.getWalkerLogLikelihood(args, flow, wsteps, cmb=False)
betterwalkerparams = walks.extractBetterWalkerStart(s2, ll2)
wsteps2 = walks.rewalkWalkers(betterwalkerparams, nsteps2)
s3, ll3 = walks.getWalkerLogLikelihood(args, flow, wsteps2, cmb=False)
samples, loglikelihoods = walks.getAllWalkersAndLLikelihoods(s, s2, s3, ll, ll2, ll3)

print("done")

# %%


def plot3x1D(s, ll, npoints1):
    limits = [0.001, 0.5, 0.9999]
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    for ivs, vs in enumerate(["A", "W", "N"]):
        samples = s[ivs * npoints1 : (ivs + 1) * npoints1, 0]
        likelihood = nf.exp(ll[ivs * npoints1 : (ivs + 1) * npoints1])
        ax[ivs].plot(samples, likelihood)
        ax[ivs].set_xscale("log")
        quantiles = corner.core.quantile(samples, limits, weights=likelihood)
        for q in quantiles:
            ax[ivs].axvline(q, c="k", alpha=0.5, lw=0.5)
        ax[ivs].set_title(
            f"{vs} \n [{quantiles[0]:.2f},{quantiles[1]:.2f},{quantiles[-1]:.2f}]"
        )
    return fig, ax


def plot3D(s, ll, **kwargs):
    return corner.corner(
        s,
        weights=nf.exp(ll),
        labels=["A", r"$\nu_{\rm rms}$", r"$\nu_{\rm min}$"],
        show_titles=True,
        hist_kwargs={"density": True},
        levels=[1 - np.exp(-0.5), 1 - np.exp(-2)],
        **kwargs,
    )


# %%

ranges = (
    [(0.9, 1.1), (19, 21), (65, 70)]
    if cosmicdawn
    else [(0.5, 1.5), (13, 15), (16.0, 16.8)]
)
truths = [1.0, 20.0, 67.5] if cosmicdawn else [1.0, 14.0, 16.4]
# plot3D(samples,loglikelihoods,range=ranges,plot_datapoints=True,bins=100,truths=truths)
plt.show()

lname = nf.get_lname(args, plot="all")
print(f"saving corner likelihood results to {lname}")
breakpoint()
np.savetxt(
    lname,
    np.column_stack([samples, loglikelihoods]),
    header="amp,width,numin,loglikelihood",
)
breakpoint()


# # %%
