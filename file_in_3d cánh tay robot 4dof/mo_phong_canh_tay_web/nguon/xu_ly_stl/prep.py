import numpy as np, json, gzip, struct, time, sys
import trimesh, fast_simplification
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SC='/tmp/claude-0/-home-claude/0dba2a93-2c79-53e0-9666-92e42ee4cbb8/scratchpad/'
V=np.load(SC+'V.npy').astype(np.float64); F=np.load(SC+'F.npy'); fl=np.load(SC+'fl.npy')
nc=fl.max()+1
# proper per-component bbox from all corners
lo=np.full((nc,3),1e9); hi=np.full((nc,3),-1e9); cnt=np.bincount(fl,minlength=nc)
for k in range(3):
    fv=V[F[:,k]]
    for ax in range(3):
        np.minimum.at(lo[:,ax],fl,fv[:,ax]); np.maximum.at(hi[:,ax],fl,fv[:,ax])
ctr=(lo+hi)/2; sz=hi-lo

# ---------- pivots in CAD coords (X, Z) ----------
P=dict(S=(63.2,215.7),E=(145.0,329.3),TL=(115.0,341.3),TR=(160.0,349.3),apex=(122.0,362.0),
       P24=(33.2,227.7),P45=(40.4,248.2),W1=(225.7,214.9),W2=(240.7,234.9))

# ---------- group assignment ----------
G={}   # id -> group name
def put(group,ids):
    for i in ids: G[i]=group
put('lower',[50,106,145,162,195,107])
put('crank',[51,43,109,135,164,194,45,59,60])
put('rodA',[13,14])
put('elbow',[228,229,258,259,260,261,262,267,226,227,234,235,294,295,300,301])
put('forearm',[245,242])
put('rodB',[33])
put('rodC',[292,293])
put('wrist',[314,315,316,317,320,333,334,339,350,351,352,353,354,355,356,357,360,361])
put('fingerL',[358,367,365,364])
put('fingerR',[359,363,368,366])
put('pinion',[362])
put('pinA',[113]); put('pinB',[116])

pivot_groups=[('E','elbow'),('TL','elbow'),('TR','elbow'),('apex','forearm'),('P45','crank'),('W1','wrist'),('W2','wrist')]
def assign(i):
    if i in G: return G[i]
    cx,cy,cz=ctr[i]
    if cz<136.5 or (cx<34.5 and cz<156) or i==81: return 'base'
    for nm,g in pivot_groups:
        px,pz=P[nm]
        if np.hypot(cx-px,cz-pz)<14: return g
    if cx>245 and 195<cz<245: return 'wrist'
    return 'turret'

PLASTIC={26,50,51,43,106,13,14,33,245,292,293,228,229,314,350,52,7,8,358,359,363,367,364,366}
MOTOR={2,66,67,357}
NAMES={
 2:('Động cơ bước NEMA 17 – xoay thân','Quay toàn bộ thân robot quanh trục thẳng đứng (J1).'),
 66:('Động cơ bước vai trái','Dẫn động bánh răng tay đòn dưới (J2) qua bánh răng nhỏ.'),
 67:('Động cơ bước vai phải','Dẫn động bánh răng tay quay khuỷu (J3) qua bánh răng nhỏ.'),
 357:('Động cơ bước tay kẹp','Quay bánh răng nhỏ, đẩy 2 thanh răng để mở/đóng ngón kẹp.'),
 52:('Bàn xoay (turntable)','Đĩa quay nằm trên ổ bi chặn, mang toàn bộ thân robot.'),
 26:('Tấm hông thân xoay','Tấm khung chính của thân; có “tai” giữ chốt cố định P24 của thanh song song.'),
 50:('Bánh răng vai A + tay đòn dưới','Bánh răng lớn gắn cứng với tay đòn dưới 140 mm (J2).'),
 106:('Tay đòn dưới (Lower arm)','Dài 140 mm từ vai S đến khuỷu E. Góc của nó là J2.'),
 51:('Bánh răng vai B + tay quay','Bánh răng thứ hai, cứng với tấm tay quay 43 – điều khiển cẳng tay (J3).'),
 43:('Tay quay khuỷu (crank)','Tấm có chốt P45, truyền chuyển động tới cẳng tay qua thanh truyền.'),
 13:('Thanh song song 1 (cam)','Cùng tay đòn dưới tạo hình bình hành → tấm khuỷu luôn giữ nguyên hướng.'),
 14:('Thanh song song 1 (cam)','Cùng tay đòn dưới tạo hình bình hành → tấm khuỷu luôn giữ nguyên hướng.'),
 33:('Thanh truyền cẳng tay','Từ chốt P45 lên đuôi cẳng tay; quyết định góc cẳng tay.'),
 245:('Cẳng tay (Forearm)','Dài 140 mm từ khuỷu E đến cổ tay W. Quay quanh E.'),
 292:('Thanh song song 2 (giữ kẹp)','Cùng cẳng tay tạo hình bình hành thứ hai → tay kẹp luôn thẳng đứng.'),
 293:('Thanh song song 2 (giữ kẹp)','Cùng cẳng tay tạo hình bình hành thứ hai → tay kẹp luôn thẳng đứng.'),
 228:('Tấm khuỷu','Dịch chuyển song song (không xoay) khi tay đòn dưới nâng hạ.'),
 229:('Tấm khuỷu','Dịch chuyển song song (không xoay) khi tay đòn dưới nâng hạ.'),
 314:('Khối cổ tay','Giữ nguyên hướng nằm ngang nhờ hình bình hành, mang tay kẹp.'),
 350:('Mâm gắn tay kẹp','Mâm phẳng gắn động cơ và cơ cấu kẹp.'),
 358:('Ngón kẹp trái','Trượt dọc trục Y nhờ thanh răng.'),
 359:('Ngón kẹp phải','Trượt dọc trục Y nhờ thanh răng.'),
 363:('Thanh răng phải','Ăn khớp bánh răng nhỏ, kéo ngón phải.'),
 367:('Thanh răng trái','Ăn khớp bánh răng nhỏ, kéo ngón trái.'),
 362:('Bánh răng nhỏ tay kẹp','Quay khi mở/đóng, truyền chuyển động cho 2 thanh răng đối xứng.'),
 113:('Bánh răng chủ động (pinion) A','Gắn trục động cơ trái, ăn khớp bánh răng lớn.'),
 116:('Bánh răng chủ động (pinion) B','Gắn trục động cơ phải, ăn khớp bánh răng lớn.'),
 7:('Khung đế','Khung đỡ cố định, đặt trên mặt bàn.'),
 8:('Thanh cảm biến / cữ chặn','Gắn vào thân xoay.'),
}
# ---------- coordinate transform: CAD(X,Y,Z) -> three (X-63.2, Z-80.87, -(Y-71.4)) ----------
def to3(p): 
    return np.stack([p[:,0]-63.2, p[:,2]-80.87, -(p[:,1]-71.4)],1)

# ---------- decimation budget ----------
def budget(i,n,grp):
    if i in (50,51): return 9000
    if i in (113,116): return 3500
    if i in (66,67,2): return 6500
    if grp=='base': r,cap=0.30,4500
    else: r,cap=0.45,6000
    if i in PLASTIC: cap=max(cap,4000)
    return int(min(n,max(min(n,150), min(cap, n*r))))

def crease_normals(v,f,angle_deg=38):
    m=trimesh.Trimesh(v,f,process=False)
    adj=m.face_adjacency; ang=m.face_adjacency_angles
    keep=ang<np.radians(angle_deg)
    a=adj[keep]
    nf=len(f)
    g=coo_matrix((np.ones(len(a)),(a[:,0],a[:,1])),shape=(nf,nf))
    nlab,lab=connected_components(g,directed=False)
    fn=m.face_normals; ar=m.area_faces
    key=(f.reshape(-1).astype(np.int64))*nlab+np.repeat(lab,3)
    uk,inv=np.unique(key,return_inverse=True)
    nv=len(uk)
    orig=(uk//nlab).astype(np.int64)
    acc=np.zeros((nv,3))
    w=(fn*ar[:,None])
    for k in range(3):
        np.add.at(acc,inv.reshape(-1,3)[:,k],w)
    l=np.linalg.norm(acc,axis=1,keepdims=True); l[l==0]=1
    return v[orig], inv.reshape(-1,3), acc/l


GL={'base':'đế cố định','turret':'thân xoay','lower':'tay đòn dưới','crank':'tay quay khuỷu','rodA':'thanh song song 1','elbow':'tấm khuỷu','forearm':'cẳng tay','rodB':'thanh truyền cẳng tay','rodC':'thanh song song 2','wrist':'cổ tay','fingerL':'ngón kẹp trái','fingerR':'ngón kẹp phải','pinion':'bánh răng kẹp','pinA':'bánh răng chủ động A','pinB':'bánh răng chủ động B'}
NEAR={'S':'khớp vai S','E':'khớp khuỷu E','TL':'chốt TL của tấm khuỷu','TR':'chốt TR của tấm khuỷu','apex':'đỉnh đuôi cẳng tay','P24':'chốt cố định P24','P45':'chốt P45 của tay quay','W1':'cổ tay W','W2':'chốt W2 cổ tay'}
EXTRA={'trục xoay thân':(63.2,141.5),'trục động cơ vai':(63.2,178.7),'cụm tay kẹp':(279.8,210.0)}
def auto_name(i):
    d=sz[i]; kind=None
    for ax in range(3):
        o=[d[k] for k in range(3) if k!=ax]
        if abs(o[0]-o[1])<=0.08*max(o)+0.15:
            dia=(o[0]+o[1])/2; l=d[ax]
            if dia<=12 and l>=1.6*dia: kind='Vít / chốt'
            elif l<=0.45*dia: kind='Ổ bi / vòng đệm' if dia>=12 else 'Vòng đệm'
            elif dia<=12: kind='Đầu vít / đai ốc'
            else: kind='Trục / ổ bi'
            nm=f'{kind} Ø{dia:.1f} × {l:.1f} mm'; break
    else:
        a,b,c=sorted(d,reverse=True); nm=f"{'Chi tiết nhỏ' if a<25 else 'Chi tiết'} {a:.0f}×{b:.0f}×{c:.0f} mm"
    cx,cz=ctr[i][0],ctr[i][2]
    cand=[(np.hypot(cx-P[k][0],cz-P[k][1]),NEAR[k]) for k in NEAR]+[(np.hypot(cx-v[0],cz-v[1]),k) for k,v in EXTRA.items()]
    dist,where=min(cand)
    ds=f"Thuộc {GL.get(assign(i),'robot')}, " + (f"gần {where} (cách {dist:.0f} mm)." if dist<28 else "nằm trên thân khâu.") + " Loại chi tiết được suy ra từ hình dạng và kích thước đo trên mô hình."
    return nm,ds

t0=time.time()
parts=[]; blobs=[]; off=0
merged={}  # (group,mat) -> lists
tot_in=tot_out=0; dropped=0
order=np.argsort(-cnt)
for i in order:
    n=cnt[i]
    if sz[i].max()<3.0: dropped+=n; continue
    grp=assign(i)
    mat='motor' if i in MOTOR else ('plastic' if i in PLASTIC else 'steel')
    fm=np.where(fl==i)[0]
    ff=F[fm]
    used,inv=np.unique(ff,return_inverse=True)
    v=V[used]; f=inv.reshape(-1,3)
    tgt=budget(i,n,grp)
    if tgt<n:
        try:
            v2,f2=fast_simplification.simplify(v.astype(np.float32),f.astype(np.int32),target_count=tgt,agg=5)
            v,f=v2.astype(np.float64),f2.astype(np.int64)
        except Exception as e:
            print('simplify fail',i,e)
    if len(f)==0: continue
    v3=to3(v)
    # restore handedness winding is preserved (proper rotation)
    vv,ii,nn=crease_normals(v3,f)
    tot_in+=n; tot_out+=len(f)
    if i in NAMES: nm,ds=NAMES[i]
    else: nm,ds=auto_name(i)
    parts.append(dict(cid=int(i),group=grp,mat=mat,v=vv,i=ii,n=nn,tris=len(f),orig=int(n),name=nm,desc=ds,dims=[round(float(x),1) for x in sz[i]]))
# đánh số các chi tiết cùng tên
from collections import Counter as _C
_cn=_C(p['name'] for p in parts); _ix={}
for p in parts:
    if _cn[p['name']]>1 and p['cid'] not in NAMES:
        _ix[p['name']]=_ix.get(p['name'],0)+1; p['name']=f"{p['name']} · {_ix[p['name']]}/{_cn[p['name']]}"
print('tris in',tot_in,'out',tot_out,'dropped tiny',dropped,'time',time.time()-t0)

# ---------- pack ----------
chunks=[]; meta=[]; o=0
def add(arr):
    global o
    b=arr.tobytes()
    pad=(-len(b))%4
    chunks.append(b+b'\0'*pad); s=o; o+=len(b)+pad; return s,len(b)
for p in parts:
    v=p['v']; mn=v.min(0); mx=v.max(0); sc=np.maximum(mx-mn,1e-6)
    q=np.round((v-mn)/sc*65535).astype(np.uint16)
    nq=np.clip(np.round(p['n']*127),-127,127).astype(np.int8)
    nq4=np.concatenate([nq,np.zeros((len(nq),1),np.int8)],1)
    idx=p['i'].reshape(-1)
    idt=np.uint16 if len(v)<=65535 else np.uint32
    po,_=add(q); no,_=add(nq4); io,_=add(idx.astype(idt))
    meta.append(dict(cid=p['cid'],group=p['group'],mat=p['mat'],name=p['name'],desc=p['desc'],
        nv=int(len(v)),ni=int(len(idx)),i32=(idt==np.uint32),mn=[float(x) for x in mn],sc=[float(x) for x in sc],po=po,no=no,io=io,
        tris=int(p['tris']),orig=p['orig'],dims=p.get('dims',[])))
blob=b''.join(chunks)
man=json.dumps(dict(parts=meta,pivots={k:[v[0]-63.2,v[1]-80.87] for k,v in P.items()},stats=dict(orig_tris=int(cnt.sum()),dropped=int(dropped),final=int(tot_out+0))),ensure_ascii=False).encode()
man+=b' '*((-len(man))%4)
raw=struct.pack('<I',len(man))+man+blob
gz=gzip.compress(raw,9)
open('/home/claude/build/robot.bin.gz','wb').write(gz)
print('raw MB',len(raw)/1e6,'gz MB',len(gz)/1e6,'parts',len(meta))
from collections import Counter
c=Counter(); 
for m in meta: c[m['group']]+=m['tris']
print(dict(c))
