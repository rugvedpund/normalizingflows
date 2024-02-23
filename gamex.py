from scipy import *
import random
import scipy.linalg as la

class Sample:
    def __init__ (self,pars, like, like0, glike):
        self.pars=pars
        self.like=like
        self.glike=glike
        self.like0=like0
        
class Gaussian:
    def __init__(self,mean,cov):
        self.cov=cov
        self.mean=mean
        self.chol=la.cholesky(cov)
        self.lndet=log(self.chol.diagonal()).sum()*2.0
        self.icov=la.inv(cov)
        self.N=len(cov)

    def sample(self):
        da=array([random.gauss(0.,1.) for x in range(self.N)])
        glike = -(da**2).sum()/2.0-self.lndet/2.0
        sa=dot(da,self.chol)
        if (self.mean is not None):
            sa+=self.mean
        return sa,glike
    
    def chi2(self,vec):
        if mean!=None:
            delta=vec-self.mean
        else:
            delta=vec
        return dot(dot(delta,self.icov),delta)
        
    def like(self,vec):
        return -self.chi2(vec)/2-self.lndet/2.0
        

class Game:
    def __init__ (self, likefunc, par0, sigreg=0.0):
        random.seed(10)
        self.like=likefunc ## returns log like
        self.sigreg=array(sigreg)
        self.N=len(par0)
        self.N1=1000
        self.blow=2.0 ## factor by which to increase the enveloping Gauss
        self.tweight=2.00
        self.wemin=0.00
        self.mineffsamp=self.N1*1
        self.fixedcov=False
        self.par0=par0

    def run(self):
        done=False
        toexplore=array(self.par0)
        badlist=[]
        Gausses=[]
        SamLists=[]
        while not done:
            sample_list, G=self.isample (toexplore)
            Gausses.append(G)
            SamLists.append(sample_list)

            toexplore=self.rebuild_samples(SamLists, Gausses)
            
            if (self.wemax<self.tweight):
                done=True
            if (len(Gausses)==20):
                print("Max iter exceeded")
                done=True
            if (self.effsamp<self.mineffsamp):
                done=False
            
        ## nothing to do here, last one is done.
        self.Gausses=Gausses

    def gausses_eval(self,Gs,pars):
        probi=0.0
        for Gx in Gs:
            probi+=exp(Gx.like(pars))
        return probi

    def rebuild_samples(self, SamLists,Gausses):
        #likes=array([[sa.like for sa in sl] for sl in SamLists])
        #maxlike= likes.flatten().max()
        maxlike=-1e30
        for sl in SamLists:
            for sa in sl:
                if (sa.like>maxlike):
                    maxlike=sa.like
                    maxlikepars=sa.pars

        gmaxlike=self.gausses_eval(Gausses,maxlikepars)
        wemax=0.0
        flist=[]
        wemax=0.0
        parmaxw=None
        effsamp=0
        for sl in SamLists:
            for sa in sl:
                rellike=exp(sa.like-maxlike)
                glike=self.gausses_eval(Gausses,sa.pars)/gmaxlike
                we=rellike/glike
                sa.we=we
                effsamp+=min(1.0,we)
                if we>wemax:
                    wemax=we
                    parmaxw=sa.pars
                if we>self.wemin:
                    flist.append(sa)
                
        self.sample_list=flist
        print("#G=",len(Gausses), "maxlike=",maxlike,"wemax=",wemax,"effsamp=",effsamp)
        self.effsamp=effsamp
        self.wemax=wemax
        return parmaxw



    def prune_badlist(self,badlist,gausses):
        stop("NOT WORKING")
        newbadlist=[]
        worstwex=0.0
        todo=None
        for G,sa in badlist:
            ## let's see what weight would it get somewhere else:
            ok=False
            wex=1e10
            for Gx in gausses:
                wex=min(wex,exp((sa.like-Gx.like0-Gx.like(sa.pars))))
                if wex<self.tweight:
                    ok=True
                    break
            sa.wex=wex
            if not ok:
                newbadlist.append((G,sa))
                if (worstwex<wex):
                    worstwex=wex
                    todo=sa.pars

        print("Pruning:", len(newbadlist),len(badlist))
        print("New Todo:",todo)
        return newbadlist,todo
                        
    def getcov(self, around):
        N=self.N

        if (self.fixedcov):
            cov=zeros((N,N))
            for i in range(N):
                cov[i,i]=self.sigreg[i]**2
            print(cov)
            G=Gaussian(around,cov)    
            return G

        icov=zeros((N,N))
        delta=self.sigreg/1000.0
        like0=self.like(around)
        for i in range(N):
            parspi=around*1.0
            parsmi=around*1.0
            parspi[i]+=delta[i]
            parsmi[i]-=delta[i]
            for j in range(N):
                if (i==j):
                    der=(self.like(parspi)+self.like(parsmi)-2*like0)/(delta[i]**2)
                else:
                    parspp=parspi*1.0
                    parspm=parspi*1.0
                    parsmp=parsmi*1.0
                    parsmm=parsmi*1.0
                    parspp[j]+=delta[j]
                    parspm[j]-=delta[j]
                    parsmp[j]+=delta[j]
                    parsmm[j]-=delta[j]
                    #print parspp, parsmm, parsmp, parspm
                    der=(self.like(parspp)+self.like(parsmm)-self.like(parspm)-self.like(parsmp))/(4*delta[i]*delta[j])
                    #print der,self.like(parspp),self.like(parsmm),self.like(parspm),self.like(parsmp)
                icov[i,j]=-der
                icov[j,i]=-der
        while True:
            print("Regularizing cholesky")
            for i in range(N):
                icov[i,i]+=1/self.sigreg[i]**2
            try:
                ch=la.cholesky(icov)
                break
            except:
                pass

        cov=la.inv(icov)
        print(cov)
        G=Gaussian(around,self.blow*cov)    
        return G


    def isample (self, zeropar):
        like0=self.like(zeropar)
        
        ## Get local covariance matrix
        G=self.getcov(zeropar)
        G.like0=like0
        slist=[]
        ## now sample around this 
        for i in range(self.N1):
            par,glike=G.sample()
            like=self.like(par)
            slist.append(Sample(par,like, like0, glike))

        return slist,G
