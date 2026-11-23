"""Sinh ảnh viên thuốc con nhộng tổng hợp có đáp án (số viên biết trước)."""
import numpy as np, json, sys, os
from PIL import Image, ImageFilter
O=sys.argv[1]; N=int(sys.argv[2]); seed=int(sys.argv[3]); os.makedirs(O, exist_ok=True)
rng=np.random.RandomState(seed)
H,W=360,520
COLORS=[(205,35,30),(230,90,20),(40,90,190),(240,200,40),(30,150,80),(235,235,230),(150,40,140)]
def capsule_mask(cx,cy,L,R,ang,yy,xx):
    ux,uy=np.cos(ang),np.sin(ang)
    dx,dy=xx-cx,yy-cy
    t=dx*ux+dy*uy; s=-dx*uy+dy*ux          # dọc thân, ngang thân
    tc=np.clip(t,-L/2,L/2)
    d=np.hypot(t-tc,s)                       # khoảng cách tới trục
    return d<=R, s/R, t
labels=[]
for i in range(N):
    yy,xx=np.mgrid[0:H,0:W].astype(float)
    bgc=np.array([rng.randint(225,255)]*3)+rng.randint(-8,8,3)
    img=np.ones((H,W,3))*bgc
    shadow=np.zeros((H,W))
    n=rng.randint(4,11); placed=[]; tries=0
    two=rng.rand()<0.5
    while len(placed)<n and tries<400:
        tries+=1
        R=rng.uniform(16,24); L=R*rng.uniform(2.0,3.2)
        cx,cy=rng.uniform(R+L/2,W-R-L/2),rng.uniform(R+L/2*0.6,H-R-L/2*0.6)
        ang=rng.uniform(0,np.pi)
        # cho phép chạm/chồng nhẹ: tâm không quá gần nhau
        if any(np.hypot(cx-p[0],cy-p[1])<1.1*R+0.3*min(L,p[3]) for p in placed): continue
        placed.append((cx,cy,R,L,ang))
    for (cx,cy,R,L,ang) in placed:
        m,_,_=capsule_mask(cx+5,cy+7,L,R*1.05,ang,yy,xx); shadow=np.maximum(shadow,m*0.18)
    sh=np.array(Image.fromarray((shadow*255).astype('uint8')).filter(ImageFilter.GaussianBlur(5)))/255
    img*=(1-sh[...,None])
    c1=np.array(COLORS[rng.randint(len(COLORS))],float); c2=np.array(COLORS[rng.randint(len(COLORS))],float) if two else c1
    for (cx,cy,R,L,ang) in placed:           # vẽ lần lượt -> viên sau đè viên trước
        m,s,t=capsule_mask(cx,cy,L,R,ang,yy,xx)
        shade=np.sqrt(np.clip(1-s**2,0,1))   # khối trụ: sáng giữa, tối mép
        base=np.where((t>0)[...,None],c1,c2)
        col=base*(0.35+0.65*shade[...,None])
        hl=np.exp(-((s+0.35)/0.12)**2)*0.8  # vệt bóng sáng
        col=col+(255-col)*hl[...,None]
        img[m]=col[m]
    img+=rng.normal(0,3,img.shape)
    im=Image.fromarray(np.clip(img,0,255).astype('uint8')).filter(ImageFilter.GaussianBlur(0.8))
    f=f'thuoc_{seed}_{i:02d}.png'; im.save(O+f); labels.append(dict(file=f,so_luong=len(placed)))
json.dump(labels,open(O+f'labels_{seed}.json','w'))
print([l['so_luong'] for l in labels])
