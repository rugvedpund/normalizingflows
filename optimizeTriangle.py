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
parser.add_argument('--sigma', type=float, default=2.0, required=False)
parser.add_argument('--subsample_factor', type=int, default=None, required=False)
parser.add_argument('--chromatic', action='store_true')
parser.add_argument('--galcut', type=float, default=20.0, required=False)
parser.add_argument('--noPCA', action='store_true')

parser.add_argument('--combineSigma', type=str, required=False, default='') #e.g. '4 6' separated by space

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

root=os.environ['LUSEE_ML']

#load ulsa map
fg=fitsio.read(f'{root}200.fits')

#try loading
fname=nf.get_fname(args)
flow=nf.FlowAnalyzerV2(nocuda=False,loadPath=fname)
flow.set_fg(args)

print('setting t21...')
freqs=np.arange(1,51)
t21=lusee.MonoSkyModels.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=16.4,A=0.04)
flow.set_t21(t21, include_noise=args.noisyT21)
if args.retrain: flow.train(flow.train_data, flow.validate_data, nocuda=False, savePath=fname,retrain=True)

# npoints=1000
# # kwargs={'amin':0.0,'amax':150.0,'wmin':1.0,'wmax':40.0,'nmin':1.0,'nmax':40.0}
# # kwargs={'amin':0.0,'amax':3.0,'wmin':10.0,'wmax':20.0,'nmin':10.0,'nmax':20.0}
# # kwargs={'amin':0.5,'amax':1.5,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0}
# kwargs={'amin':0.01,'amax':100.0,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0,
#          'logspace':True}

#3D corner
npoints=50
kwargs={'amin':0.9,'amax':1.1,'wmin':13.9,'wmax':14.1,'nmin':16.35,'nmax':16.45}
# kwargs={'amin':0.5,'amax':1.5,'wmin':13.5,'wmax':14.5,'nmin':16.0,'nmax':16.8}
# kwargs={'amin':0.01,'amax':2.0,'wmin':13.0,'wmax':15.00,'nmin':15.4,'nmax':17.4}
# kwargs={'amin':0.0,'amax':50.0,'wmin':10.0,'wmax':18.0,'nmin':12.4,'nmax':20.4}
attempt=0
done=False
while done==False:
    print('getting 3d likelihood...')
    print(kwargs)
    samples,t21_vs=nf.get_t21vs(npoints,**kwargs)
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
    if attempt>10: 
        print('too many attempts, breaking')
        done=True


#save
lname=nf.get_lname(args,plot='all')

for arg in vars(args):
    if getattr(refargs,arg)!=getattr(args,arg): print("==>",BOLD,RED,arg,getattr(args,arg),END)
    else: print(arg, getattr(args, arg))
    
# corner.corner(samples,weights=nf.exp(likelihood),bins=50,
#             labels=['Amplitude','Width',r'$\nu_{min}$'], truths=[1.0,14.0,16.4],
#             verbose=True, plot_datapoints=False, show_titles=True,
#             levels=[1-np.exp(-0.5),1-np.exp(-2)])
# # plt.suptitle(f'{lname.split("/")[-1]}')
# plt.show()


print(f'saving corner likelihood results to {lname}')
np.savetxt(lname,np.column_stack([samples,likelihood]),header='amp,width,numin,loglikelihood')
