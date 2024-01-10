import pandas as pd
import NormalizingFlow as nf

args=nf.Args()

fgfreqs=[('ulsa.fits','1 51'),('gsm16.fits','51 101')]
chromatics=[True,False]
snrpps=[1e4, 1e5,1e6,1e7,1e8]
seeds=[0,1,2,3,4,5,6,7,8,9]
combineSigmas=['','4','4 6']
versus=['A','all']

df=list()
for fg,freq in fgfreqs:
    for chromatic in chromatics:
        for snrpp in snrpps:
            for seed in seeds:
                for cs in combineSigmas:
                    for vs in versus:
                        args.fgFITS=fg
                        args.freqs=freq
                        args.chromatic=chromatic
                        args.SNRpp=snrpp
                        args.noiseSeed=seed
                        print(fg,chromatic,f'{snrpp:.0e}',seed,cs,vs)
                        try:
                            samples,loglikelihood=nf.get_samplesAndLikelihood(args,vs)
                        except FileNotFoundError:
                            print('not found, continuing')
                            continue
                        for s,ll in zip(samples,loglikelihood): 
                            df.append({
                                'fg':fg,
                                'chromatic':chromatic,
                                'SNRpp':snrpp,
                                'noiseSeed':seed,
                                'vs':vs,
                                'sample':s,
                                'loglikelihood':ll
                                })
print('done, saving...')
data=pd.DataFrame(df)
data.to_parquet('likelihoods')
print('done')
print(data)

