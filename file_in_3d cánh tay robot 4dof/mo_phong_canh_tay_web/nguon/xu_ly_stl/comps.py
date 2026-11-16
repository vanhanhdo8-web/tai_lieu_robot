import numpy as np
V=np.load('V.npy'); F=np.load('F.npy'); fl=np.load('fl.npy')
nc=fl.max()+1
cnt=np.bincount(fl,minlength=nc)
vl=np.zeros(len(V),int); vl[F[:,0]]=fl
lo=np.full((nc,3),1e9);hi=np.full((nc,3),-1e9);sm=np.zeros((nc,3))
for k in range(3):
    np.minimum.at(lo[:,k],vl,V[:,k]); np.maximum.at(hi[:,k],vl,V[:,k])
np.savez('comps.npz',lo=lo,hi=hi,cnt=cnt)
import sys
x0=float(sys.argv[1]); 
sel=[i for i in range(nc) if (lo[i,0]+hi[i,0])/2>x0]
for i in sorted(sel,key=lambda i:-cnt[i]):
    print(i,cnt[i],np.round(lo[i],1),np.round(hi[i],1))
