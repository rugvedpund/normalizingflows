import numpy as np
import fitsio
import NormalizingFlow as nf
import lusee
import argparse
import os
import corner
import matplotlib.pyplot as plt

#muust provide --noisyT21 and --diffCombineSigma!!!

parser = argparse.ArgumentParser(description="Normalizing Flow for ULSA maps")
parser.add_argument('--fgFITS', type=str, required=False, default='gsm16.fits')
parser.add_argument('--sigma', type=float, default=2.0, required=False)
parser.add_argument('--subsample_factor', type=int, default=None, required=False)
parser.add_argument('--chromatic', action='store_true')
parser.add_argument('--galcut', type=float, default=20.0, required=False)
parser.add_argument('--noPCA', action='store_true')

parser.add_argument('--combineSigma', type=str, required=False, default='') #e.g. '4 6' separated by space

parser.add_argument('--SNRpp', type=float, default=None, required=False)
parser.add_argument('--noise', type=float, default=0.0, required=False)
parser.add_argument('--noiseSeed', type=int, default=0, required=False)
parser.add_argument('--subsampleSigma', type=float, default=2.0, required=False)

parser.add_argument('--noisyT21', action='store_true')
parser.add_argument('--gainFluctuationLevel', type=float, default=0.0, required=False)
parser.add_argument('--gFdebug', type=int, default=0, required=False)
parser.add_argument('--append', type=str, default='_SVD')

parser.add_argument('--DA_factor', type=float, required=False, default=1.0)
parser.add_argument('--plot', type=str, default='all', required=False) # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
parser.add_argument('--freqFluctuationLevel',type=float, required=False, default=0.0)
parser.add_argument('--nPCA', type=str, default='') #e.g 'nmin nmax' separated by space

parser.add_argument('--diffCombineSigma',action='store_true') #bool, whether to do fg_cS[sigma]-fg_cS[sigma-1]
parser.add_argument('--avgAdjacentFreqBins',action='store_true') #bool, whether to average adjacent freq bins

parser.add_argument('--retrain', action='store_true')
parser.add_argument('--appendLik', type=str, default='', required=False)
args=parser.parse_args()

refargs=nf.Args()

#interesting hack to print bold text in terminal
BOLD = '\033[1m'
RED='\033[91m'
END='\033[0m'
for arg in vars(args):
    if getattr(refargs,arg)!=getattr(args,arg): print("==>",BOLD,RED,arg,getattr(args,arg),END)
    else: print(arg, getattr(args, arg))
    
###-------------------------------------------------------------------------------------------------###

fname=nf.get_fname(args)
flow=nf.FlowAnalyzerV2(nocuda=False,loadPath=fname)
flow.set_fg(args)

print('setting t21...')
freqs=np.linspace(51,100)
t21=lusee.MonoSkyModels.T_CosmicDawn_Scaled(freqs,nu_rms=20,nu_min=67.5,A=0.130)
flow.set_t21(t21, include_noise=args.noisyT21)
if args.retrain: flow.train(flow.train_data, flow.validate_data, nocuda=False, savePath=fname,retrain=True)

#3D corner
npoints=50
kwargs={'amin':0.95,'amax':1.05,'wmin':19.0,'wmax':21.0,'nmin':66.5,'nmax':68.5}

attempt=0
done=False
while done==False:
    print('getting 3d likelihood...')
    print(kwargs)
    samples,t21_vs=nf.get_t21vs(npoints,cosmicdawn=True,**kwargs)
    t21_vsdata=flow.proj_t21(t21_vs,include_noise=True)
    likelihood=flow.get_likelihood(t21_vsdata,args.freqFluctuationLevel,args.DA_factor, debugfF=False)
    constraints=nf.get_constraints(samples,likelihood)
    print(constraints)
    for p in ['amp','width','numin']:
        print(f"{p}: {constraints[p+'+-']}",kwargs[p[0]+'max']-kwargs[p[0]+'min'])
    print('------------------')
    #update kwargs
    oldkwargs=kwargs.copy()
    for p in ['amp','width','numin']:
        if constraints[p+'+-']>0.6*(kwargs[p[0]+'max']-kwargs[p[0]+'min']):
            print(f"min/max too small, updating {p}...")
            kwargs[p[0]+'min']-=2*constraints[p+'-']
            kwargs[p[0]+'max']+=2*constraints[p+'+']
            #keep min >=0.0
            if kwargs[p[0]+'min']<0.0: 
                kwargs[p[0]+'min']=1e-12
                # done=True
                # print('min<0.0, breaking')
        
        if constraints[p+'+-']<0.2*(kwargs[p[0]+'max']-kwargs[p[0]+'min']):
            print(f"min/max too big, updating {p}...")
            newwidth=(kwargs[p[0]+'max']-kwargs[p[0]+'min'])/2.0/2.0
            kwargs[p[0]+'min']+=newwidth
            kwargs[p[0]+'max']-=newwidth
            #keep max-min >0.0
            if kwargs[p[0]+'max']-kwargs[p[0]+'min']<0.0: 
                print('This should never happen: max-min<0.0')
                raise ValueError
    
    #check if done
    if kwargs==oldkwargs:
        print('done!')
        done=True
    #break if too many attempts
    print('attempt#:',attempt)
    attempt+=1
    if attempt>0: 
        print('too many attempts, breaking')
        done=True
#save
lname=nf.get_lname(args,plot='all')
for arg in vars(args):
    if getattr(refargs,arg)!=getattr(args,arg): print("==>",BOLD,RED,arg,getattr(args,arg),END)
    else: print(arg, getattr(args, arg))
print(f'saving corner likelihood results to {lname}')
np.savetxt(lname,np.column_stack([samples,likelihood]),header='amp,width,numin,loglikelihood')


#1D
npoints=2000
# kwargs={'amin':0.01,'amax':100.0,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0,
        #   'logspace':True}
kwargs={'amin':0.01,'amax':1000.0,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0,
         'logspace':True}
for vs in ['A']:
    print(f'getting 1d likelihood for {vs}...')
    samples1d,t21_vs1d=nf.get_t21vs1d(npoints,vs,cosmicdawn=True,**kwargs)
    t21vsdata1d=flow.proj_t21(t21_vs1d,include_noise=True)
    likelihood1d=flow.get_likelihood(t21vsdata1d,args.freqFluctuationLevel,args.DA_factor,debugfF=False)
    lname=nf.get_lname(args,plot=vs)
    print(f'saving 1d likelihood results to {lname}')
    np.savetxt(lname,np.column_stack([samples1d,likelihood1d]),header='amp,width,numin,loglikelihood')

# #plot
# corner.corner(samples,weights=nf.exp(likelihood),bins=50,
#             labels=['Amplitude','Width',r'$\nu_{min}$'], truths=[1.0,20.0,67.5],
#             verbose=True, plot_datapoints=False, show_titles=True,
#             levels=[1-np.exp(-0.5),1-np.exp(-2)])
# # plt.suptitle(f'{lname.split("/")[-1]}')
# plt.show()


# s,ll=nf.get_samplesAndLikelihood(args,plot='A')
# quantiles=corner.core.quantile(s[:,0],[0.05,0.5,0.95],weights=nf.exp(ll))
# for x in quantiles:
#     plt.axvline(130*x,c='k',alpha=0.5,lw=0.5)
# plt.axvline(130,color='k')
# plt.plot(130*s,nf.exp(ll))
# plt.xscale('log')
# # plt.xlim(10,2e3)
# plt.title(f'Amplitude 95% confidence: <{130*quantiles[-1]:.3f} mK')
# plt.show()
