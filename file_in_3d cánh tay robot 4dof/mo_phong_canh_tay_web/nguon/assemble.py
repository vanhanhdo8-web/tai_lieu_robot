import base64,sys,os
# Chạy: python3 assemble.py inline ../robot_arm_web.html   (đọc file ngay cạnh script này)
here=os.path.dirname(os.path.abspath(__file__))
rd=lambda n,m='r':open(os.path.join(here,n),m,**({'encoding':'utf-8'} if m=='r' else {})).read()
t=rd('template.html')
b64=base64.b64encode(rd('robot.bin.gz','rb')).decode()
inline=sys.argv[1]=='inline'
if inline:
    libs='<script>'+rd('three.min.js')+'</script>\n<script>'+rd('OrbitControls.js')+'</script>'
else:
    libs='<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>\n<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>'
out=t.replace('/*LIBS*/',libs).replace('/*MODEL_B64*/',b64)
open(sys.argv[2],'w',encoding='utf-8').write(out)
print(sys.argv[2],len(out)/1e6,'MB')
