import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
t=time.time()
raw=open('/home/claude/extract/mo_phong_canh_tay/robot_arm_full.stl','rb').read()
n=int.from_bytes(raw[80:84],'little')
dt=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
a=np.frombuffer(raw,dtype=dt,count=n,offset=84)
tri=a['v'].reshape(-1,3).copy()   # (3n,3)
print('tris',n,'bbox',tri.min(0),tri.max(0))
# weld
q=np.round(tri/1e-3).astype(np.int64)
u,inv=np.unique(q,axis=0,return_inverse=True)
inv=inv.ravel()
print('unique verts',len(u),time.time()-t)
F=inv.reshape(-1,3)
V=(u*1e-3).astype(np.float32)
np.save('V.npy',V); np.save('F.npy',F.astype(np.int32))
# connected components over vertices
e=np.concatenate([F[:,[0,1]],F[:,[1,2]]])
g=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(V),len(V)))
nc,lab=connected_components(g,directed=False)
print('components',nc)
fl=lab[F[:,0]]
cnt=np.bincount(fl,minlength=nc)
order=np.argsort(-cnt)
np.save('fl.npy',fl)
for i in order[:60]:
    m=fl==i
    vv=V[np.unique(F[m])]
    print(i,cnt[i],np.round(vv.min(0),1),np.round(vv.max(0),1))
print('small comps (<100 tris):',(cnt<100).sum())
