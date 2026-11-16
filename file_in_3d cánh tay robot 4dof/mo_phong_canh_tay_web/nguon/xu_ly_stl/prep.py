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
def loc(i):
    cx,cz=ctr[i][0],ctr[i][2]
    cand=[(np.hypot(cx-P[k][0],cz-P[k][1]),NEAR[k]) for k in NEAR]+[(np.hypot(cx-v[0],cz-v[1]),k) for k,v in EXTRA.items()]
    dist,where=min(cand)
    return f"Thuộc {GL.get(assign(i),'robot')}, " + (f"gần {where} (cách {dist:.0f} mm)." if dist<28 else "nằm trên thân khâu.")

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
    return nm,loc(i)+" Loại chi tiết được suy ra từ hình dạng và kích thước đo trên mô hình."


# ================= ĐỐI CHIẾU VỚI BOM CỦA NGƯỜI DÙNG =================
BOMQ={'M2 10mm':6,'M3 6mm':36,'M4 10mm':6,'M4 16mm':8,'M4 20mm':6,'M6 20mm':1,'M6 40mm':2,'M6 50mm':1,
      'M2 Nut':6,'M3 Nut':4,'M4 Nut':14,'M6 Nut':4,'F624':6,'F686':12,'51105':1,
      'Nema17 Stepper Motor':3,'End Stop':3,'2GT Pulley 20 Teeth':3,'2GT-6mm closed belt 200mm':3}
BOM={}   # id -> dict(name, cat, kw, note, warn, uniq)
def B(ids,name,cat,kw,note,warn=False,uniq=False,key=None):
    for i in ids: BOM[i]=dict(name=name,cat=cat,kw=kw,note=note,warn=warn,uniq=uniq,key=key or name)
def screw(ids,bomname,head,shank,L,headtype):
    nom=float(bomname.split()[1].replace('mm',''))
    d=bomname.split()[0]
    ok=abs(L-nom)<1.0
    t=f"Vít {d} {headtype}: đầu Ø{head:g}, thân Ø{shank:g}, phần thân dài ≈ {L:g} mm trên mô hình. "
    t+=("Khớp BOM." if ok else f"⚠ BOM ghi {bomname}, nhưng mô hình vẽ thân dài ≈ {L:g} mm — tên lấy theo BOM (khớp đường kính / số lượng), nên kiểm tra lại khi lắp thật.")
    B(ids,bomname,'screw','vit oc screw bolt '+d,t,warn=not ok,key=bomname)
# --- vít ---
screw([21,22,30,44,48,49],'M2 10mm',4.5,2.0,10,'đầu trụ thấp')
M3L=[104,105,196,197,368,98,99,100,101,208,209,210,211,214,215,216,217,365]
screw([360,361],'M3 6mm',5.7,3.0,6,'đầu tròn')
screw(M3L,'M3 6mm',5.7,3.0,8,'đầu tròn')
screw([9,10,68,69,109,135,164,194],'M3 6mm',5.5,3.0,10,'đầu trụ')
screw([107,145,162,195],'M3 6mm',5.5,3.0,20,'đầu trụ')
screw([88,89,199,200,222,223],'M4 10mm',7.8,4.0,13.6,'đầu tròn dẹt (button)')
screw([95,96,204,206,219,221],'M4 16mm',7.0,4.0,16,'đầu trụ lục giác chìm')
screw([300,301,339],'M4 16mm',7.6,4.0,16,'đầu tròn dẹt (button)')
screw([59,234,235,351,352,353],'M4 20mm',7.6,4.0,20,'đầu tròn dẹt (button)')
screw([132],'M6 20mm',10,6,20,'đầu lục giác'); BOM[132]['uniq']=True
screw([320],'M6 40mm',10,6,40,'đầu lục giác')
screw([133],'M6 40mm',10,6,60,'đầu lục giác')
screw([267],'M6 50mm',10,6,50,'đầu lục giác'); BOM[267]['uniq']=True
# 133 và 320 cùng là "M6 40mm" -> đánh số
# --- đai ốc ---
B([19,20,29,42,90,91],'M2 Nut','nut','dai oc nut','Đai ốc M2 (đo 5,0 × 2,0 mm). Khớp BOM theo số lượng (6 chiếc) và vị trí nằm ngay cạnh vít M2.')
M4N=[34,35,60,92,93,94,97,202,203,205,207,218,220,224,225,236,237,252,302,303,354,355,356]
B(M4N,'M4 Nut','nut','dai oc nut m4','Đai ốc M4 (đo 7,0 mm giữa hai mặt × 3,2 mm dày — đúng chuẩn M4). Mô hình có 23 chiếc, BOM ghi 14.')
B([122,262,317],'M6 Nut','nut','dai oc nut m6','Đai ốc M6 có bích tròn Ø14,2 × 7,3 mm, lỗ Ø5. Mô hình có 3 chiếc, BOM ghi 4.')
# --- ổ bi ---
B([45,226,227,294,295,333],'F624ZZ (4×13×5)','bearing','o bi bearing f624 624',
  'Ổ bi bích F624ZZ: lỗ Ø4 × thân Ø13 × dày 5 mm, bích ngoài Ø15 mm. BOM: F624 ×6 (mô hình có 6 ổ rời + 4 ổ gộp liền vít = 10).',key='F624')
B([24,25,334],'F624ZZ (4×13×5) + M4 16mm','bearing','o bi bearing f624 624 vit m4 16',
  'Ổ bi bích F624ZZ (4×13×5 mm, bích Ø15) đã gộp liền với vít M4 16 mm xuyên qua tâm (STL dính thành một khối).',key='F624')
B([242],'F624ZZ (4×13×5) + M4 20mm','bearing','o bi bearing f624 624 vit m4 20',
  'Ổ bi bích F624ZZ (4×13×5 mm, bích Ø15) đã gộp liền với vít M4 20 mm xuyên qua tâm (STL dính thành một khối).',key='F624')
F686O=[114,115,117,118,119,120,121,258,259,260,261,315,316]
F686I=[136,137,138,139,140,141,142,272,273,274,275,323,324]
B(F686O,'F686ZZ (6×13×5) – vòng ngoài','bearing','o bi bearing f686 686',
  'Vòng ngoài ổ bi bích F686ZZ (lỗ 6 × thân Ø13 × dày 5 mm, bích Ø15). BOM: F686 ×12, mô hình có 13 ổ.',key='F686')
B(F686I,'F686ZZ (6×13×5) – vòng trong','bearing','o bi bearing f686 686',
  'Vòng trong ổ bi F686ZZ (lỗ Ø6, ngoài Ø9, dày 5 mm) – nằm trong vòng ngoài cùng vị trí.',key='F686')
B([81],'51105 (25×42×11) – vòng trục','bearing','o bi chan bearing thrust 51105',
  'Vòng trục (lỗ Ø25) của ổ chặn 51105: lỗ 25 × ngoài Ø42 × cao 11 mm. Ổ đỡ toàn bộ thân xoay.',uniq=True,key='51105')
B([82],'51105 (25×42×11) – vòng giữ bi','bearing','o bi chan bearing thrust 51105',
  'Vòng/cage giữ 16 viên bi Ø6 của ổ chặn 51105 (25×42×11 mm).',uniq=True,key='51105')
B([83],'51105 (25×42×11) – vòng vỏ','bearing','o bi chan bearing thrust 51105',
  'Vòng vỏ (lỗ Ø26) của ổ chặn 51105: lỗ 25–26 × ngoài Ø42 × cao 11 mm.',uniq=True,key='51105')
B([86,87,102,103,108,110,123,124,166,169,192,193,198,201,212,213],'Bi cầu Ø6 (51105)','bearing','bi cau o bi chan 51105 ball',
  'Viên bi thép Ø6 mm của ổ chặn 51105 – có 16 viên xếp vòng trong cage.')
# --- động cơ / puly / end stop ---
MOT={2:'J1 – xoay thân',66:'J2 – vai trái',67:'J3 – vai phải'}
for i,r in MOT.items():
    B([i],f'Nema17 Stepper Motor ({r})','motor','dong co motor nema17 stepper',
      f'Động cơ bước Nema17 (thân ≈ 42 × 42 × 46 mm) – {r}. BOM: Nema17 ×3.',uniq=True,key='Nema17 Stepper Motor')
PUL={12:'J1 – xoay thân',113:'J2 – vai trái',116:'J3 – vai phải'}
for i,r in PUL.items():
    B([i],f'2GT Pulley 20 Teeth ({r})','pulley','puly pulley 2gt banh dai',
      f'Puly 2GT 20 răng, lỗ trục Ø5 mm, bích Ø≈17 – gắn trên trục động cơ {r}. BOM: 2GT Pulley 20T ×3 đi cùng 3 dây đai 2GT-6mm vòng kín 200 mm (mô hình CAD không vẽ dây đai).',uniq=True,key='2GT Pulley 20 Teeth')
B([17,78,23],'End Stop','endstop','end stop cong tac hanh trinh',
  'Công tắc hành trình (End Stop) dày 5,8 mm – dùng để về home. BOM: End Stop ×3.')
# --- chi tiết nhận dạng được nhưng không có trong BOM ---
B([3,4,5,6,70,71,72,73,74,75,76,77],'Chân tiếp điểm Ø0,5 × 4,5','other','chan pin tiep diem',
  'Chân kim loại rất mảnh (Ø0,5 × 4,5 mm), xếp thành hàng gần động cơ. Suy ra từ hình dạng. Không có trong BOM.')
B([364,366],'Miếng đệm đầu ngón kẹp (2 mm)','other','dem ngon kep pad',
  'Miếng đệm phẳng dày 2 mm ở đầu ngón kẹp (suy ra từ hình dạng). Không có trong BOM.')
B([15,16,150,151,152,153],'Vòng đệm Ø5/6,8 × 0,9','other','vong dem washer',
  'Vòng đệm mỏng (lỗ Ø5, ngoài Ø6,8, dày 0,9 mm) trên trục động cơ. Không có trong BOM.')
B([18,158,159],'Trục động cơ Ø5 × 64,8','other','truc shaft dong co',
  'Trục động cơ Ø5 mm dài 64,8 mm, nối puly 2GT. Không có trong BOM.')
B([11,111,112],'Nhãn dán động cơ','other','nhan label sticker',
  'Miếng nhãn dán mỏng 0,1 mm trên thân động cơ Nema17. Không có trong BOM.')
B([0,1,38,39,62,63,64,65,188,189,190,191],'Góc thân Nema17 (Ø20 × 47,3)','motor','dong co motor nema17 goc',
  'Phần góc của thân động cơ Nema17 — nằm đúng lưới lỗ bắt 31 mm của động cơ (12 chi tiết = 4 góc × 3 động cơ). Không có trong BOM.')

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
    cat='inferred'; warn=False; kw=''
    if i in BOM:
        b=BOM[i]; nm=b['name']; ds=(NAMES[i][1]+' ' if i in NAMES else '')+loc(i)+' '+b['note']; cat=b['cat']; warn=b['warn']; kw=b['kw']
        if b['key'] in BOMQ: ds+=f" [BOM: {b['key']} ×{BOMQ[b['key']]}]"
    elif i in NAMES: nm,ds=NAMES[i]; cat='main'
    else: nm,ds=auto_name(i)
    parts.append(dict(cid=int(i),group=grp,mat=mat,v=vv,i=ii,n=nn,tris=len(f),orig=int(n),name=nm,desc=ds,cat=cat,warn=warn,kw=kw,bk=(BOM[i]['key'] if i in BOM else ''),bomuniq=(i in BOM and BOM[i]['uniq']),dims=[round(float(x),1) for x in sz[i]]))
# đánh số các chi tiết cùng tên
from collections import Counter as _C
_cn=_C(p['name'] for p in parts); _ix={}
for p in parts:
    if _cn[p['name']]>1 and p['cid'] not in NAMES and not p['bomuniq']:
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
        tris=int(p['tris']),orig=p['orig'],dims=p.get('dims',[]),cat=p['cat'],warn=bool(p['warn']),kw=p['kw'],bk=p['bk']))
blob=b''.join(chunks)
from collections import Counter as _C2
_f=_C2()
for p in parts:
    if p['bk'] and 'vòng trong' not in p['name'] and not p['name'].startswith('Bi cầu') and not (p['bk']=='51105' and 'vòng trục' not in p['name']): _f[p['bk']]+=1
_f['M4 16mm']+=3; _f['M4 20mm']+=1   # vít gộp liền ổ bi F624
BN={'M3 6mm':'12 vít đo dài 10–20 mm (⚠)','M4 10mm':'khớp theo số lượng (thân đo ≈ 13,6 mm)','M4 Nut':'đo đúng chuẩn M4 7×3,2','M3 Nut':'không thấy trên mô hình',
    'F624':'6 rời + 4 gộp liền vít M4','F686':'tính theo vòng ngoài; mỗi ổ có thêm vòng trong','M6 40mm':'1 chiếc đo dài 60 mm (⚠)','2GT-6mm closed belt 200mm':'CAD không vẽ dây đai',
    'M2 Nut':'khớp theo số lượng (đo 5×2 mm)','M6 Nut':'bích tròn Ø14,2','51105':'3 vòng + 16 bi Ø6'}
bomtab=[[k,q,int(_f.get(k,0)),BN.get(k,'')] for k,q in BOMQ.items()]
man=json.dumps(dict(bom=bomtab,parts=meta,pivots={k:[v[0]-63.2,v[1]-80.87] for k,v in P.items()},stats=dict(orig_tris=int(cnt.sum()),dropped=int(dropped),final=int(tot_out+0))),ensure_ascii=False).encode()
man+=b' '*((-len(man))%4)
raw=struct.pack('<I',len(man))+man+blob
gz=gzip.compress(raw,9)
open('/home/claude/build/robot.bin.gz','wb').write(gz)
print('raw MB',len(raw)/1e6,'gz MB',len(gz)/1e6,'parts',len(meta))
from collections import Counter
c=Counter(); 
for m in meta: c[m['group']]+=m['tris']
print(dict(c))
