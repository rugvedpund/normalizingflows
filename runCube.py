import numpy as np
import torch
import fitsio
import NormalizingFlow as nf
import lusee
import parser
import os
import corner
import matplotlib.pyplot as plt


##---------------------------------------------------------------------------##
# argparser block

parser = parser.create_parser()
args = parser.parse_args()

# must have --noisyT21 and --diffCombineSigma!!!
args.noisyT21 = True
args.diffCombineSigma = True

args.appendLik = "_cube"

args.print()

##---------------------------------------------------------------------------##

# set seed
print("setting noise and torch seeds...")
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
flow.set_t21(t21, include_noise=args.noisyT21)
if args.retrain:
    flow.train(
        flow.train_data, flow.validate_data, nocuda=False, savePath=fname, retrain=True
    )

# start with 1D amplitude plot
print("getting 1D amplitude plot...")
kwargs = {
    "amin": 0.001,
    "amax": 100.0,
    "wmin": 1.0,
    "wmax": 50.0,
    "nmin": 1.0,
    "nmax": 50.0,
    "logspace": True,
}
if cosmicdawn:
    kwargs["nmin"] = 50.0
    kwargs["nmax"] = 100.0
print(kwargs)

samples1d, t21_vs1d = nf.get_t21vs1d(
    flow.freqs, npoints=2000, vs="A", cosmicdawn=cosmicdawn, **kwargs
)
t21_vsdata1d = flow.proj_t21(t21_vs1d, include_noise=True)
likelihood1d = flow.get_likelihood(
    t21_vsdata1d, args.freqFluctuationLevel, args.DA_factor
)
lname = nf.get_lname(args, plot="A")
print(f"saving 1d likelihood to {lname}")
np.savetxt(
    lname,
    np.column_stack([samples1d, likelihood1d]),
    header="amp,width,numin,loglikelihood",
)

limits = [0.001, 0.5, 0.9999]
s, ll = nf.get_samplesAndLikelihood(args, plot="A")
quantiles = corner.core.quantile(s[:, 0], limits, weights=nf.exp(ll))
print("quantiles:", quantiles)

# #plot amp
# for x in quantiles:
#     plt.axvline(x,c='k',alpha=0.5,lw=0.5)
# plt.axvline(1.0,color='k')
# plt.plot(s,nf.exp(ll))
# plt.xscale('log')
# # plt.xlim(10,2e3)
# plt.title(f'{quantiles=}')
# plt.show()

if quantiles[0] < 0.01:
    print("skipping 3D corner plot")
    exit()

truths = [1.0, 20.0, 67.5] if cosmicdawn else [1.0, 14.0, 16.4]

print("getting 1D limits on width and numin...")
for ivs, vs in enumerate(["W", "N"]):
    samples1d, t21_vs1d = nf.get_t21vs1d(
        flow.freqs, npoints=2000, vs=vs, cosmicdawn=cosmicdawn, **kwargs
    )
    t21_vsdata1d = flow.proj_t21(t21_vs1d, include_noise=True)
    likelihood1d = flow.get_likelihood(
        t21_vsdata1d, args.freqFluctuationLevel, args.DA_factor
    )
    lname = nf.get_lname(args, plot=vs)
    print(f"saving 1d likelihood to {lname}")
    np.savetxt(
        lname,
        np.column_stack([samples1d, likelihood1d]),
        header="amp,width,numin,loglikelihood",
    )

kwargs3D = {}
for ivs, vs in enumerate(["A", "W", "N"]):
    s, ll = nf.get_samplesAndLikelihood(args, plot=vs)
    quantiles = corner.core.quantile(s[:, 0], limits, weights=nf.exp(ll))
    print("quantiles:", quantiles)
    kwargs3D[vs.lower() + "min"] = max(
        truths[ivs] - 2 * (quantiles[-1] - truths[ivs]), 0.01
    )
    kwargs3D[vs.lower() + "max"] = truths[ivs] + 2 * (quantiles[-1] - truths[ivs])

    # #plot
    # for x in quantiles:
    #     plt.axvline(x,c='k',alpha=0.5,lw=0.5)
    # plt.plot(s,nf.exp(ll))
    # plt.xscale('log')
    # # plt.xlim(10,2e3)
    # plt.title(f'{quantiles=}')
    # plt.show()


# kwargs3D={'amin':0.5,'amax':1.5,'wmin':13.5,'wmax':14.5,'nmin':16.0,'nmax':16.8}
print(kwargs3D)

print("getting 3D corner plot...")
samples, t21_vs = nf.get_t21vs(
    flow.freqs, npoints=30, cosmicdawn=cosmicdawn, **kwargs3D
)
t21_vsdata = flow.proj_t21(t21_vs, include_noise=True)
likelihood = flow.get_likelihood(
    t21_vsdata, args.freqFluctuationLevel, args.DA_factor, debugfF=False
)
lname = nf.get_lname(args, plot="all")
print(f"saving corner likelihood results to {lname}")
np.savetxt(
    lname,
    np.column_stack([samples, likelihood]),
    header="amp,width,numin,loglikelihood",
)

# plot
corner.corner(
    samples,
    weights=nf.exp(likelihood),
    bins=30,
    labels=["Amplitude", "Width", r"$\nu_{min}$"],
    truths=[1.0, 20.0, 67.5] if cosmicdawn else [1.0, 14.0, 16.4],
    verbose=True,
    plot_datapoints=False,
    show_titles=True,
    levels=[1 - np.exp(-0.5), 1 - np.exp(-2)],
)
# plt.suptitle(f'{lname.split("/")[-1]}')
plt.show()
