import gamex
import sys,random
import matplotlib.pyplot as plt
import numpy as np
import NormalizingFlow as nf
import corner

random.seed(100)

if sys.argv[1]=='gauss':
    def like(x):
        x=np.array(x).reshape(-1,3)
        # c2=np.array([[1,0.9],
        #             [0,2]])
        c=np.array([[3,1,1],
                    [0,4,1],
                    [0,0,2]])
        return np.diag(-(x@c@x.T)/2.0)
    ga=gamex.Game(like,[0.5,0.0,0.0],[0.7,0.7,0.7])
    ga.N1=1000
    ga.tweight=1.50
    ga.mineffsamp=5000
    sname='gauss.pdf'
    ga.run()

elif sys.argv[1]=='ring':
    def like(x):
        x=np.array(x)
        r2=(x**2).sum(axis=1 if x.ndim==2 else None)
        return -(r2-4.0)**2/(2*0.5**2)
    ga=gamex.Game(like,[3.5,0.0,0.0],[0.3,0.3,0.3])
    ga.blow=2.0
    ga.tweight=1.50
    sname='ring.pdf'
    ga.run()

if sys.argv[1]=='dblgauss':
    def like(x):
        x=np.array(x)
        # c2=np.array([[1,0.9],
        #             [0,2]])
        c1=np.array([[1,1,1],
                    [0,2,1],
                    [0,0,3]])
        c2=0.3*c1
        m2=np.array([3,3,3])
        return -(x@c1@x.T + (x-m2)@c2@(x-m2).T)/2.0 #centered at 0,0,0 and m2
    ga=gamex.Game(like,[4.5,0.0,0.0],[0.7,0.7,0.7])
    ga.N1=1000
    ga.tweight=1.50
    ga.mineffsamp=5000
    sname='gauss.pdf'
    ga.run()

elif sys.argv[1]=='box':
    def like(x):
        if (abs(x).any()>1):
            return -30
        else:
            return 0
    ga=Game(like,[0.5,0.0,0.0],[0.4,0.4,0.4])
    ga.tweight=1.4
    ga.N1=10
    ga.run()
    sname='box.pdf'

def plotel(G):
    global fig
    cov=G.cov
    print(G.cov)
    val,vec=np.linalg.eig(cov)
    vec=vec.T

    vec[0]*=np.sqrt(np.real(val[0]))
    vec[1]*=np.sqrt(np.real(val[1]))
    print(vec[0],'A')
    print(vec[1],'B')
    corner.overplot_points(fig,G.mean[None],c='b',marker='o')
    corner.overplot_points(fig,[G.mean-vec[0],
                                G.mean+vec[0]],c='r',marker='',linestyle='-')
    corner.overplot_points(fig,[G.mean-vec[1],
                                G.mean+vec[1]],c='r',marker='',linestyle='-')
    corner.overplot_points(fig,[G.mean-vec[2],
                                G.mean+vec[2]],c='r',marker='',linestyle='-')


## now we plot
xyz=np.array([sa.pars for sa in ga.sample_list])
ww=np.array([sa.we for sa in ga.sample_list]).flatten()

cornerkwargs={'show_titles':True,'plot_contours':False,'bins':100,'range':[(-5,5),(-5,5),(-5,5)],
        'no_fill_contours':True, 'labels':['x','y','z'], 'pcolor_kwargs':{'cmap':'viridis'}}

# just samples
fig=corner.corner(xyz,**cornerkwargs)
plt.suptitle('All Samples')
plt.show()

# weighted samples, with gaussians
wsumsa=ww/ww.sum()
fig=corner.corner(xyz,weights=wsumsa,**cornerkwargs)
for G in ga.Gausses:
    plotel(G)
plt.suptitle('Weighted Samples')
plt.show()

# true values
trvals=np.exp(like(xyz))
trvalsa=trvals/trvals.sum()
fig=corner.corner(xyz,weights=trvalsa,**cornerkwargs)
plt.suptitle('trvals')
plt.show()

# diffp
diffp=wsumsa-trvalsa
fig=corner.corner(xyz,weights=diffp,**cornerkwargs)
plt.suptitle('diffp')
plt.show()

breakpoint()



