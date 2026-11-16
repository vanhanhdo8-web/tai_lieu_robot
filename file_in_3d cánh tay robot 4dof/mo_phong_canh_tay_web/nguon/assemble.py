import base64,sys
t=open('/home/claude/build/template.html',encoding='utf-8').read()
b64=base64.b64encode(open('/home/claude/build/robot.bin.gz','rb').read()).decode()
inline=sys.argv[1]=='inline'
if inline:
    libs='<script>'+open('/home/claude/build/three.min.js').read()+'</script>\n<script>'+open('/home/claude/build/OrbitControls.js').read()+'</script>'
else:
    libs='<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>\n<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>'
out=t.replace('/*LIBS*/',libs).replace('/*MODEL_B64*/',b64)
open(sys.argv[2],'w',encoding='utf-8').write(out)
print(sys.argv[2],len(out)/1e6,'MB')
