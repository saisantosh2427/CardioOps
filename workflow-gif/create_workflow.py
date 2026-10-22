from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
OUT=Path(__file__).resolve().parent
W,H=1200,1500
INK='#182e40'; MUTED='#526675'; GRID='#e6ebee'; BG='#fafbf9'
fontroot=Path('C:/Windows/Fonts')
def font(size,bold=False,hand=False):
    return ImageFont.truetype(str(fontroot/('comicbd.ttf' if hand else ('segoeuib.ttf' if bold else 'segoeui.ttf'))),size)
def text(d,xy,s,size=22,color=INK,bold=False,center=False,hand=False):
    f=font(size,bold,hand)
    if center:
        box=d.textbbox((0,0),s,font=f); xy=(xy[0]-(box[2]-box[0])/2,xy[1])
    d.text(xy,s,font=f,fill=color)
# Flow follows a serpentine reading order, then closes the production learning loop.
nodes=[
('UCI heart data','920 patient records','13 clinical measurements',80,300,'#dff3ff','data'),
('Validate + clean','Pandera schema checks','Types, missing values, splits',450,300,'#dff7e8','check'),
('Build features','Feature engineering','scikit-learn transformers',820,300,'#eae3ff','feature'),
('Train + compare','LR / Random Forest / GB','GridSearchCV + hold-out test',820,560,'#ffe6d5','train'),
('MLflow tracking','Parameters, metrics, models','Postgres + MinIO artifacts',450,560,'#e0efff','chart'),
('Model registry','Champion / challenger','Versioned models + rollback',80,560,'#fff0bb','registry'),
('FastAPI serving','Client requests → predictions','Loads champion automatically',80,820,'#dff7e8','api'),
('Prediction log','Predictions + actual labels','Postgres monitoring store',450,820,'#dff3ff','data'),
('Evidently monitor','Data drift + performance','Champion reference vs. traffic',820,820,'#eae3ff','chart'),
('Check thresholds','Drift or performance drop?','If healthy: keep monitoring',820,1080,'#fff0bb','check'),
('Train challenger','Learn from labeled traffic','Evaluate on production hold-out',450,1080,'#ffe6d5','train'),
('Promotion gate','Promote only if better','Otherwise retain the champion',80,1080,'#ffdfe4','registry')]
paths=[[(380,380),(450,380)],[(750,380),(820,380)],[(970,460),(970,560)],[(820,640),(750,640)],[(450,640),(380,640)],[(230,720),(230,820)],[(380,900),(450,900)],[(750,900),(820,900)],[(970,980),(970,1080)],[(820,1160),(750,1160)],[(450,1160),(380,1160)],[(80,1160),(38,1160),(38,640),(80,640)]]
colors=['#147caa']*6+['#16876c']*3+['#b35b2d']*3
captions=[
'01  Collect public heart disease data',
'02  Validate records and prepare clean data',
'03  Transform features for model training',
'04  Compare candidates and select the best model',
'05  Track experiments, metrics and artifacts',
'06  Register a champion for production serving',
'07  Serve predictions and record production traffic',
'08  Attach confirmed labels for performance checks',
'09  Monitor drift and model performance',
'10  Trigger retraining when thresholds are crossed',
'11  Evaluate the challenger against the champion',
'12  Promote only if better; serving picks up the new model']
def point(path,t):
    lengths=[math.dist(a,b) for a,b in zip(path,path[1:])]; distance=t*sum(lengths)
    for a,b,L in zip(path,path[1:],lengths):
        if distance<=L:return (a[0]+(b[0]-a[0])*distance/L,a[1]+(b[1]-a[1])*distance/L)
        distance-=L
    return path[-1]
def line(d,path,color):
    length=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
    for n in range(0,int(length)-12,17):
        d.line([point(path,n/length),point(path,min(n+9,length-12)/length)],fill=color,width=3)
    a,b=path[-2:]; angle=math.atan2(b[1]-a[1],b[0]-a[0])
    d.polygon([b,(b[0]-13*math.cos(angle-.48),b[1]-13*math.sin(angle-.48)),(b[0]-13*math.cos(angle+.48),b[1]-13*math.sin(angle+.48))],fill=color)
def icon(d,x,y,kind):
    c=INK
    if kind=='data':
        d.rectangle((x,y+7,x+42,y+33),fill='white',outline=c,width=2)
        d.ellipse((x,y,x+42,y+14),fill='white',outline=c,width=2)
        d.arc((x,y+25,x+42,y+39),0,180,fill=c,width=2)
        d.line((x,y+33,x,y+30),fill=c,width=2)
    elif kind=='check':
        d.rounded_rectangle((x+4,y,x+38,y+38),radius=6,fill='white',outline=c,width=2)
        d.line([(x+11,y+19),(x+19,y+27),(x+32,y+11)],fill='#16876c',width=4)
    elif kind=='chart':
        for i,h in enumerate([13,23,35]):d.rounded_rectangle((x+i*14,y+38-h,x+i*14+9,y+38),radius=2,fill=c)
    elif kind=='registry':
        for i in range(3):d.rounded_rectangle((x+i*3,y+26-i*12,x+38+i*3,y+37-i*12),radius=3,fill='white',outline=c,width=2)
    elif kind=='api':
        text(d,(x-3,y-8),'{ }',36,bold=True)
    elif kind=='feature':
        for i in range(3):
            d.line((x,y+7+i*13,x+42,y+7+i*13),fill=c,width=2)
            p=[13,29,20][i];d.ellipse((x+p-4,y+3+i*13,x+p+4,y+11+i*13),fill='white',outline=c,width=2)
    else:
        for a,b in [((6,8),(36,8)),((6,8),(21,34)),((36,8),(21,34))]:d.line((x+a[0],y+a[1],x+b[0],y+b[1]),fill=c,width=2)
        for a,b in [(6,8),(36,8),(21,34)]:d.ellipse((x+a-5,y+b-5,x+a+5,y+b+5),fill='white',outline=c,width=2)
def frame(k):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    for x in range(0,W,40):d.line((x,0,x,H),fill=GRID)
    for y in range(0,H,40):d.line((0,y,W,y),fill=GRID)
    d.rounded_rectangle((55,35,1145,146),radius=42,fill='#ffda52')
    text(d,(600,49),'Heart Disease MLOps',52,center=True,hand=True)
    text(d,(600,160),'FROM PATIENT DATA TO A MODEL THAT KEEPS LEARNING',22,center=True,bold=True)
    text(d,(600,201),'Train  •  Track  •  Serve  •  Monitor  •  Improve',24,center=True,color=MUTED)
    for p,c in zip(paths,colors):line(d,p,c)
    active=k//8; phase=(k%8)/8
    for i,(title,s1,s2,x,y,fill,kind) in enumerate(nodes):
        d.rounded_rectangle((x+5,y+6,x+305,y+166),radius=17,fill='#dce2e4')
        d.rounded_rectangle((x,y,x+300,y+160),radius=17,fill=fill,outline=colors[i] if i==active else INK,width=5 if i==active else 2)
        icon(d,x+20,y+18,kind)
        text(d,(x+278,y+22),f'{i+1:02}',17,color=MUTED,bold=True,center=True)
        text(d,(x+20,y+62),title,25,bold=True)
        text(d,(x+20,y+101),s1,18)
        text(d,(x+20,y+127),s2,17,color=MUTED)
    for i,path in enumerate(paths):
        q=point(path,(phase if i==active else (k/32)%1))
        r=7 if i==active else 4
        d.ellipse((q[0]-r,q[1]-r,q[0]+r,q[1]+r),fill=colors[i],outline='white',width=2)
    # Connector annotations remain outside cards.
    for x,y,label in [(970,490,'fit + evaluate'),(230,750,'load champion'),(970,1010,'evaluate'),(785,1117,'retrain')]:
        box=d.textbbox((0,0),label,font=font(16)); tw=box[2]-box[0]
        d.rounded_rectangle((x-tw/2-7,y-2,x+tw/2+7,y+23),radius=5,fill=BG)
        text(d,(x,y),label,16,color=MUTED,center=True)
    d.rounded_rectangle((70,1300,1130,1392),radius=19,fill=INK)
    text(d,(600,1315),captions[active],23,color='white',center=True,bold=True)
    for i in range(12):
        x=278+i*55;d.rounded_rectangle((x,1363,x+39,1369),radius=3,fill='#ffda52' if i<=active else '#526675')
    text(d,(600,1413),'Docker Compose  •  scikit-learn  •  MLflow  •  FastAPI  •  Evidently',19,center=True,color=MUTED)
    text(d,(600,1450),'Reference project using public UCI data • Not for clinical decisions',16,center=True,color=MUTED)
    return im
frames=[frame(i) for i in range(96)]
frames[0].save(OUT/'mlops-workflow.png')
palette=frames[0].quantize(colors=128)
frames=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
frames[0].save(OUT/'mlops-workflow.gif',save_all=True,append_images=frames[1:],duration=125,loop=0,optimize=False,disposal=1)
print(OUT/'mlops-workflow.gif')
