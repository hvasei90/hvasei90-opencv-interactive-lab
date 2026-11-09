"""OpenCV Interactive Lab — real Python/OpenCV processing for teaching."""
from __future__ import annotations
import base64, io, math
from typing import Any
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from pathlib import Path

ROOT=Path(__file__).parent
app=FastAPI(title='OpenCV Interactive Lab')
app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')

class Payload(BaseModel):
    topic: str
    image: str | None = None
    params: dict[str,Any] = Field(default_factory=dict)
    points: list[list[int]] = Field(default_factory=list)


def sample() -> np.ndarray:
    h,w=640,960
    y,x=np.mgrid[:h,:w]
    a=np.zeros((h,w,3),dtype=np.uint8)
    a[:,:,0]=np.clip(35+x*.11,0,255).astype(np.uint8)
    a[:,:,1]=np.clip(24+y*.13,0,255).astype(np.uint8)
    a[:,:,2]=np.clip(27+x*.038+y*.028,0,255).astype(np.uint8)
    cv2.rectangle(a,(90,125),(350,410),(50,180,240),-1,cv2.LINE_AA)
    cv2.circle(a,(645,255),123,(230,190,60),-1,cv2.LINE_AA)
    pts=np.array([[450,480],[585,360],[720,470],[840,445],[785,565],[520,568]],np.int32)
    cv2.fillPoly(a,[pts],(160,75,190))
    cv2.line(a,(80,530),(880,100),(245,240,240),9,cv2.LINE_AA)
    cv2.putText(a,'OPEN CV',(310,90),cv2.FONT_HERSHEY_SIMPLEX,1.85,(255,255,255),4,cv2.LINE_AA)
    for i in range(22):
        cv2.circle(a,(60+i*39,600),3,(205,205,205),-1)
    return a

def encode(a):
    ok,b=cv2.imencode('.png',a)
    if not ok: raise ValueError('Image encoding failed')
    return 'data:image/png;base64,'+base64.b64encode(b).decode()

def decode(value):
    if not value: return sample()
    try:
        raw=base64.b64decode(value.split(',')[-1],validate=True)
        if len(raw)>12_000_000: raise ValueError('تصویر بیش از حد بزرگ است.')
        arr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if arr is None: raise ValueError('فرمت تصویر قابل خواندن نیست.')
        if arr.shape[0]*arr.shape[1]>8_000_000: raise ValueError('ابعاد تصویر بیش از حد بزرگ است.')
        return arr
    except Exception as exc: raise ValueError('تصویر ورودی نامعتبر است.') from exc

# Each resulting statement is executed as-is by Python. Only image acquisition and
# returning the encoded PNG are handled by the HTTP layer outside the shown code.
def build(topic:str, p:dict, points:list, image:np.ndarray):
    def num(name,default,low,high,integer=True):
        try: v=float(p.get(name,default))
        except (TypeError,ValueError):v=float(default)
        v=min(high,max(low,v))
        return int(round(v)) if integer else round(v,3)
    k=num('kernel',5,1,31)|1
    t=num('threshold',125,0,255)
    th=num('thickness',8,1,45)
    scale=num('scale',70,20,180)
    angle=num('angle',35,-180,180)
    alpha=num('alpha',.55,0,1,False)
    c=num('c',4,-25,25)
    block=num('block',11,3,51)|1
    low=num('low',65,0,255)
    high=num('high',150,0,255)
    amount=num('amount',55,-250,250)
    def point(i, fallback):
        if len(points)>i and len(points[i])==2:
            return (max(0,min(image.shape[1]-1,int(points[i][0]))),max(0,min(image.shape[0]-1,int(points[i][1]))))
        return fallback
    w=image.shape[1];h=image.shape[0]
    a=point(0,(int(w*.18),int(h*.25)))
    b=point(1,(int(w*.78),int(h*.73)))
    color=p.get('color','#ff9933')
    if not isinstance(color,str) or len(color)!=7 or any(v not in '0123456789abcdefABCDEF' for v in color[1:]): color='#ff9933'
    bgr=tuple(int(color[i:i+2],16) for i in (5,3,1))
    # CV source code has no unsafe user expression interpolation: all values sanitized above.
    C=[]
    add=C.append
    add('import cv2')
    add('import numpy as np')
    add('')
    add('# image: decoded BGR input selected in the lab')
    add('result = image.copy()')
    if topic=='imread':add('# Actual image loading is handled by cv2.imdecode in the API');add('result = image.copy()')
    elif topic=='imwrite':add("success, png = cv2.imencode('.png', image)");add('result = cv2.imdecode(png, cv2.IMREAD_COLOR)')
    elif topic=='imshow':add('# Browser preview replaces cv2.imshow (desktop GUI only)');add('result = image.copy()')
    elif topic in ('waitKey','destroyAllWindows','keyboard','mouseCallback','mouseEvents'):
        add('# Browser keyboard / mouse events are captured by the UI')
        add('result = image.copy()')
    elif topic=='pixel':
        add(f'x, y = {a!r}')
        add('result[y, x] = (0, 255, 255)')
        add('cv2.drawMarker(result, (x,y), (0,0,255), cv2.MARKER_CROSS, 22, 2)')
    elif topic.startswith('np_'):
        if topic=='np_shape':
            add('shape, ndim, dtype = image.shape, image.ndim, str(image.dtype)')
            add('summary = pd.DataFrame([{"shape": str(shape), "ndim": ndim, "dtype": dtype, "height": shape[0], "width": shape[1], "channels": shape[2]}])')
        elif topic=='np_pixel':
            add(f'x, y = {a!r}')
            add('pixel = image[y, x].copy()')
            add('summary = pd.DataFrame([{"x": x, "y": y, "B": int(pixel[0]), "G": int(pixel[1]), "R": int(pixel[2])}])')
            add('cv2.drawMarker(result, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)')
        elif topic=='np_setpixel':
            radius=num('blocksize',35,5,151)//2
            add(f'x, y = {a!r}')
            add(f'radius = {radius}')
            add('y1, y2 = max(0, y-radius), min(image.shape[0], y+radius+1)')
            add('x1, x2 = max(0, x-radius), min(image.shape[1], x+radius+1)')
            add(f'result[y1:y2, x1:x2] = {bgr!r}')
        elif topic=='np_slice':
            add(f'(x1, y1), (x2, y2) = {a!r}, {b!r}')
            add('result = image[min(y1,y2):max(y1,y2)+1, min(x1,x2):max(x1,x2)+1].copy()')
        elif topic=='np_channels':
            ch=str(p.get('channel','B')).upper()
            channel={'B':0,'G':1,'R':2}.get(ch,0)
            add(f'channel = {channel}  # B=0, G=1, R=2')
            add('result = image[:, :, channel].copy()')
        elif topic=='np_where':
            add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)')
            add(f'mask = gray > {t}')
            add('result = np.where(mask[:, :, None], image, 0).astype(np.uint8)')
        elif topic=='np_clip':
            add(f'amount = {num("amount",40,-150,150)}')
            add('result = np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)')
        elif topic=='np_flip':add('result = np.flip(image, axis=1).copy()')
        elif topic=='np_rot90':add(f'result = np.rot90(image, k={num("turns",1,1,3)}).copy()')
        elif topic=='np_stats':
            add('summary = pd.DataFrame([{"min": int(np.min(image)), "max": int(np.max(image)), "mean": round(float(np.mean(image)), 3), "sum": int(np.sum(image, dtype=np.int64))}])')
        elif topic=='np_reshape':
            add('flat = image.reshape(-1, 3)')
            add('summary = pd.DataFrame([{"original_shape": str(image.shape), "reshaped_shape": str(flat.shape), "flatten_size": int(image.flatten().size), "ravel_size": int(image.ravel().size)}])')
    elif topic.startswith('pd_'):
        add('import pandas as pd')
        if topic=='pd_head':
            add('h, w = image.shape[:2]')
            add('flat = image.reshape(-1, 3)')
            add('indices = np.arange(h*w, dtype=np.int64)')
            add('df = pd.DataFrame({"x": indices % w, "y": indices // w, "B": flat[:,0], "G": flat[:,1], "R": flat[:,2]})')
            add(f'summary = df.head({num("rows",8,3,20)})')
        elif topic=='pd_describe':
            add('df = pd.DataFrame(image.reshape(-1, 3), columns=["B", "G", "R"])')
            add('summary = df.describe().round(2).reset_index()')
        elif topic=='pd_groupby':
            add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)')
            add('level = pd.cut(gray.ravel(), bins=[-1,63,127,191,255], labels=["dark", "mid-dark", "mid-light", "light"])')
            add('df = pd.DataFrame({"level": level, "intensity": gray.ravel()})')
            add('summary = df.groupby("level", observed=True).agg(count=("intensity", "size"), mean=("intensity", "mean")).round(2).reset_index()')
        elif topic=='pd_hist':
            add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)')
            add('df = pd.DataFrame({"bin": gray.ravel() // 16})')
            add('summary = df["bin"].value_counts().sort_index().rename_axis("bin_0_to_15").reset_index(name="pixels")')
    elif topic=='resize':add(f'result = cv2.resize(image, None, fx={scale/100}, fy={scale/100}, interpolation=cv2.INTER_LINEAR)')
    elif topic=='line':add(f'result = cv2.line(result, {a!r}, {b!r}, {bgr!r}, {th}, cv2.LINE_AA)')
    elif topic=='circle':add(f'result = cv2.circle(result, {a!r}, {max(4,round(math.dist(a,b)))}, {bgr!r}, {th}, cv2.LINE_AA)')
    elif topic=='rectangle':add(f'result = cv2.rectangle(result, {a!r}, {b!r}, {bgr!r}, {th}, cv2.LINE_AA)')
    elif topic=='ellipse':add(f'result = cv2.ellipse(result, {a!r}, ({max(4,abs(b[0]-a[0]))}, {max(4,abs(b[1]-a[1]))}), {angle}, 0, 300, {bgr!r}, {th}, cv2.LINE_AA)')
    elif topic in ('polylines','fillPoly'):
        poly=points if len(points)>=3 else [[int(w*.15),int(h*.25)],[int(w*.63),int(h*.15)],[int(w*.82),int(h*.65)],[int(w*.35),int(h*.82)]]
        add(f'vertices = np.array({poly!r}, dtype=np.int32)')
        if topic=='polylines':add(f'result = cv2.polylines(result, [vertices], True, {bgr!r}, {th}, cv2.LINE_AA)')
        else:add(f'cv2.fillPoly(result, [vertices], {bgr!r})')
    elif topic=='putText':add(f"result = cv2.putText(result, 'OpenCV', {a!r}, cv2.FONT_HERSHEY_SIMPLEX, 2, {bgr!r}, {th}, cv2.LINE_AA)")
    elif topic=='crop':add(f'x1,y1 = {a!r}');add(f'x2,y2 = {b!r}');add('result = image[min(y1,y2):max(y1,y2)+1, min(x1,x2):max(x1,x2)+1].copy()')
    elif topic in ('gray','rgb','hsv','hls','lab','ycrcb'):
        const={'gray':'COLOR_BGR2GRAY','rgb':'COLOR_BGR2RGB','hsv':'COLOR_BGR2HSV','hls':'COLOR_BGR2HLS','lab':'COLOR_BGR2LAB','ycrcb':'COLOR_BGR2YCrCb'}[topic]
        add(f'converted = cv2.cvtColor(image, cv2.{const})')
        if topic=='gray':add('result = converted')
        else:
            add('# Display is normalized back to BGR for faithful screen colors')
            reverse={'rgb':'COLOR_RGB2BGR','hsv':'COLOR_HSV2BGR','hls':'COLOR_HLS2BGR','lab':'COLOR_LAB2BGR','ycrcb':'COLOR_YCrCb2BGR'}[topic]
            add(f'result = cv2.cvtColor(converted, cv2.{reverse})')
    elif topic=='split':
        channel=num('channel',0,0,2)
        add('b, g, r = cv2.split(image)');add(f'result = [b, g, r][{channel}]')
    elif topic=='merge':add('b,g,r = cv2.split(image)');add('result = cv2.merge((r,g,b))')
    elif topic=='inRange':add('hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)');add(f'result = cv2.inRange(hsv, np.array([{low//2}, 45, 40]), np.array([{max(low//2,high//2)}, 255, 255]))')
    elif topic=='colormap':
        add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)');add('result = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)')
    elif topic in ('blending','alpha','overlay'):
        add('second = cv2.flip(image, 1)')
        add(f'result = cv2.addWeighted(image, {alpha}, second, {round(1-alpha,3)}, 0)')
    elif topic in ('bitwise_and','bitwise_or','bitwise_not','bitwise_xor','mask'):
        add('mask = np.zeros(image.shape[:2], np.uint8)');add(f'cv2.circle(mask, ({w//2},{h//2}), {min(w,h)//3}, 255, -1)')
        if topic in ('mask','bitwise_and'):add('result = cv2.bitwise_and(image, image, mask=mask)')
        elif topic=='bitwise_not':add('result = cv2.bitwise_not(image)')
        else:
            add('layer = cv2.bitwise_not(image)')
            add(f'result = cv2.{topic}(image, layer)')
    elif topic=='translate':add(f'M = np.float32([[1,0,{amount}],[0,1,{round(amount*.6)}]])');add('result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))')
    elif topic=='rotate':add(f'M = cv2.getRotationMatrix2D(({w/2},{h/2}), {angle}, 1.0)');add(f'result = cv2.warpAffine(image, M, ({w},{h}))')
    elif topic=='flip':add('result = cv2.flip(image, 1)')
    elif topic=='affine':
        add(f'src = np.float32([[0,0],[{w-1},0],[0,{h-1}]])')
        add(f'dst = np.float32([[{max(0,amount)},0],[{w-1},40],[50,{h-1}]])')
        add('M = cv2.getAffineTransform(src, dst)');add(f'result = cv2.warpAffine(image, M, ({w},{h}))')
    elif topic=='perspective':
        add(f'src = np.float32([[0,0],[{w-1},0],[0,{h-1}],[{w-1},{h-1}]])')
        add(f'dst = np.float32([[{w//6},0],[{w-w//6},0],[0,{h-1}],[{w-1},{h-1}]])')
        add('M = cv2.getPerspectiveTransform(src,dst)');add(f'result = cv2.warpPerspective(image, M, ({w},{h}))')
    elif topic=='blur':add(f'result = cv2.blur(image, ({k},{k}))')
    elif topic=='gaussian':add(f'result = cv2.GaussianBlur(image, ({k},{k}), 0)')
    elif topic=='median':add(f'result = cv2.medianBlur(image, {k})')
    elif topic=='filter2d':
        matrix=p.get('matrix')
        if isinstance(matrix,list) and len(matrix)==9:
            try: values=[max(-5,min(5,float(v))) for v in matrix]
            except (ValueError,TypeError):values=[0,-1,0,-1,5,-1,0,-1,0]
        else: values=[0,-1,0,-1,5,-1,0,-1,0]
        add(f'kernel = np.array({[values[:3],values[3:6],values[6:]]!r}, dtype=np.float32)')
        add('result = cv2.filter2D(image, -1, kernel)')
    elif topic.startswith('threshold_'):
        add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)')
        mode=topic.replace('threshold_','')
        if mode in ('adaptiveMean','adaptiveGaussian'):
            meth='ADAPTIVE_THRESH_MEAN_C' if mode=='adaptiveMean' else 'ADAPTIVE_THRESH_GAUSSIAN_C'
            add(f'result = cv2.adaptiveThreshold(gray,255,cv2.{meth},cv2.THRESH_BINARY,{block},{c})')
        else:
            meth={'binary':'THRESH_BINARY','inverse':'THRESH_BINARY_INV','trunc':'THRESH_TRUNC','tozero':'THRESH_TOZERO','tozeroInv':'THRESH_TOZERO_INV','otsu':'THRESH_BINARY + cv2.THRESH_OTSU'}[mode]
            add(f'_, result = cv2.threshold(gray,{t},255,cv2.{meth})')
    elif topic in ('sobel','scharr','laplacian','canny'):
        add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)')
        if topic=='canny':add(f'result = cv2.Canny(gray, {min(low,high)}, {max(low,high)})')
        elif topic=='laplacian':add('edges = cv2.Laplacian(gray, cv2.CV_64F)');add('result = cv2.convertScaleAbs(edges)')
        elif topic=='sobel':add(f'gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={min(k,7)})');add(f'gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={min(k,7)})');add('result = cv2.convertScaleAbs(cv2.magnitude(gx.astype(np.float32),gy.astype(np.float32)))')
        else:add('gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)');add('gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)');add('result = cv2.convertScaleAbs(cv2.magnitude(gx.astype(np.float32),gy.astype(np.float32)))')
    elif topic in ('erode','dilate','opening','closing','gradient','tophat','blackhat'):
        add(f'kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({k},{k}))')
        if topic in ('erode','dilate'):add(f'result = cv2.{topic}(image,kernel)')
        else:
            op={'opening':'OPEN','closing':'CLOSE','gradient':'GRADIENT','tophat':'TOPHAT','blackhat':'BLACKHAT'}[topic]
            add(f'result = cv2.morphologyEx(image,cv2.MORPH_{op},kernel)')
    elif topic=='contours':
        add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)');add('_, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)')
        retrieval=str(p.get('retrieval','RETR_TREE'))
        if retrieval not in ('RETR_EXTERNAL','RETR_LIST','RETR_CCOMP','RETR_TREE'):retrieval='RETR_TREE'
        chain=str(p.get('chain','CHAIN_APPROX_SIMPLE'))
        if chain not in ('CHAIN_APPROX_NONE','CHAIN_APPROX_SIMPLE'):chain='CHAIN_APPROX_SIMPLE'
        add(f'contours, hierarchy = cv2.findContours(binary,cv2.{retrieval},cv2.{chain})')
        add('cv2.drawContours(result, contours, -1, (0,255,255), 3)')
        add('areas = [cv2.contourArea(cnt) for cnt in contours]')
    elif topic in ('harris','shi','subpix'):
        add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)')
        if topic=='harris':
            add('response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)')
            add('result[response > 0.015 * response.max()] = [0,0,255]')
        else:
            add('corners = cv2.goodFeaturesToTrack(gray, 80, 0.01, 10)')
            if topic=='subpix':add('criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)');add('corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria) if corners is not None else None')
            add('if corners is not None:')
            add('    for pt in corners.reshape(-1,2):')
            add('        cv2.circle(result, tuple(np.int32(np.round(pt))), 5, (0,255,255), -1)')
    elif topic=='template':
        x1,y1=a;x2,y2=b
        if abs(x2-x1)<8 or abs(y2-y1)<8:x1,y1,x2,y2=int(w*.15),int(h*.22),int(w*.45),int(h*.55)
        x1,x2=sorted((x1,x2));y1,y2=sorted((y1,y2))
        method=str(p.get('method','TM_CCOEFF_NORMED'))
        if method not in ('TM_CCOEFF','TM_CCOEFF_NORMED','TM_SQDIFF','TM_SQDIFF_NORMED'):method='TM_CCOEFF_NORMED'
        add(f'template = image[{y1}:{y2},{x1}:{x2}]')
        add(f'matching = cv2.matchTemplate(image, template, cv2.{method})')
        add('min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(matching)')
        add(f'best = min_loc if cv2.{method} in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else max_loc')
        add(f'cv2.rectangle(result, best, (best[0]+{x2-x1},best[1]+{y2-y1}), (0,255,255), 4)')
    elif topic in ('orb','sift','surf'):
        if topic=='surf': raise ValueError('SURF در اغلب بسته‌های OpenCV نصب نیست (nonfree). برای اجرای آن به بیلد اختصاصی opencv_contrib نیاز دارید.')
        add(f'detector = cv2.{"ORB_create(nfeatures=500)" if topic=="orb" else "SIFT_create(nfeatures=500)"}')
        add('keypoints, descriptors = detector.detectAndCompute(image, None)')
        add('result = cv2.drawKeypoints(image, keypoints, None, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)')
    elif topic=='haar':
        add('gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)')
        add("cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')")
        add('faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)')
        add('for x,y,fw,fh in faces:')
        add('    cv2.rectangle(result,(x,y),(x+fw,y+fh),(0,255,255),3)')
    else: raise ValueError('این دستور در نسخه فعلی پیاده‌سازی نشده است.')
    return '\n'.join(C)

@app.get('/')
def home():return FileResponse(ROOT/'static'/'index.html')

@app.get('/api/sample')
def sample_route():return {'image':encode(sample()),'width':960,'height':640}

@app.post('/api/process')
def process(body:Payload):
    try:
        image=decode(body.image)
        script=build(body.topic,body.params,body.points,image)
        space={'image':image,'cv2':cv2,'np':np,'pd':pd}
        exec(compile(script,'<lesson>','exec'),{'__builtins__':__builtins__},space)
        result=space['result']
        if result is None or not result.size:raise ValueError('خروجی این تنظیمات خالی است.')
        result=np.asarray(result)
        if result.dtype!=np.uint8:result=np.clip(result,0,255).astype(np.uint8)
        meta={'shape':list(result.shape),'bytes':len(result.tobytes())}
        if 'summary' in space:
            frame=space['summary']
            meta['table']={'columns':[str(x) for x in frame.columns], 'rows':[[round(float(v),3) if isinstance(v,(float,np.floating)) else (int(v) if isinstance(v,(int,np.integer)) else str(v)) for v in row] for row in frame.itertuples(index=False,name=None)]}
        if body.topic=='contours':meta['count']=len(space.get('contours',[]))
        if body.topic in ('orb','sift'):meta['count']=len(space.get('keypoints',[]))
        if body.topic=='haar':meta['count']=len(space.get('faces',[]))
        if result.ndim==2:result=cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
        return {'output':encode(result),'code':script,'meta':meta}
    except ValueError as exc:raise HTTPException(400,str(exc))
    except Exception as exc:raise HTTPException(400,f'اجرای دستور ناموفق بود: {type(exc).__name__}: {str(exc)[:180]}')

@app.post('/api/histogram')
def histogram(body:Payload):
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        image=decode(body.image)
        fig=Figure(figsize=(9,3.5),dpi=110,facecolor='#101b30')
        ax=fig.add_subplot(111,facecolor='#101b30')
        channels=cv2.split(image)
        for ch,col,lbl in zip(channels,['#5ba6ff','#58e0bd','#ff747e'],['B','G','R']):
            data=cv2.calcHist([ch],[0],None,[256],[0,256]).ravel()
            ax.plot(np.arange(256),data,color=col,lw=1.7,label=lbl)
        ax.tick_params(colors='#cbd5e1');ax.spines[['top','right']].set_visible(False)
        ax.spines['left'].set_color('#475569');ax.spines['bottom'].set_color('#475569')
        ax.set_xlim(0,255);ax.legend(facecolor='#1f2e48',labelcolor='white')
        fig.tight_layout()
        out=io.BytesIO();fig.savefig(out,format='png',facecolor=fig.get_facecolor());return {'output':'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode(),'code':"b, g, r = cv2.split(image)\nfor channel in (b, g, r):\n    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])\n    plt.plot(hist)"}
    except Exception as exc:raise HTTPException(400,str(exc))
