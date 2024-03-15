import numpy as np
import pandas as pd
import NormalizingFlow as nf

def get_df(args,SNRpps,seeds,combineSigmas):
    rawdata=list()
    for snrpp in SNRpps:
        for seed in seeds:
            for cs in combineSigmas:
                args.SNRpp=snrpp
                args.noiseSeed=seed
                args.combineSigma=cs
                s,ll=nf.get_samplesAndLikelihood(args,plot='A')
                rawdata.append({'snrpp':snrpp,'seed':seed,'cs':cs,'s':s.reshape(-1),'l':nf.exp( ll )})
    df=pd.DataFrame(rawdata).set_index(['snrpp','seed','cs'])
    pltdata=pd.DataFrame(columns=['s','med','std','medlow','medhigh'],
                         index=pd.MultiIndex.from_product([SNRpps,combineSigmas],
                                                          names=['snrpp','cs']))
    #create pltdata with medians and stds
    for snrpp in SNRpps:
        for cs in combineSigmas:
            pltdata['s'][snrpp,cs]=np.median([df['s'][snrpp,seed,cs] for seed in seeds],axis=0)
            pltdata['med'][snrpp,cs]=np.median([df['l'][snrpp,seed,cs] for seed in seeds],axis=0)
            pltdata['std'][snrpp,cs]=np.std([df['l'][snrpp,seed,cs] for seed in seeds],axis=0)
            pltdata['medlow'][snrpp,cs]=np.clip(pltdata['med'][snrpp,cs]-pltdata['std'][snrpp,cs],0,None)
            pltdata['medhigh'][snrpp,cs]=np.clip(pltdata['med'][snrpp,cs]+pltdata['std'][snrpp,cs],0,None)

    return pltdata
