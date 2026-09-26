/*
 * 真实 Seam Carving 后台算法 + 结果 20% 高斯模糊：
 * Sobel 能量 → 动态规划 → 回溯 seam → 删除像素。
 * 横向压缩删除纵向 seam；
 * 纵向压缩通过转置图像后复用算法。
 * 不使用普通缩放、分区拉伸或 CSS transform。
 */
function seamWorker(){
 class Carver{
 constructor(image){
 this.w=image.width;
 this.h=image.height;
 this.stride=image.width;

 const n=this.w*this.h;
 this.pixels=new Uint8ClampedArray(image.pixels);
 this.gray=new Float32Array(n);
 this.energy=new Float32Array(n);
 this.cost=new Float64Array(n);
 this.parent=new Int8Array(n);
 this.seam=new Int32Array(this.h);

 for(let i=0;i<n;i++){
 const p=i*4;
 const alpha=this.pixels[p+3]/255;
 const luminance=
 .2126*this.pixels[p]+
 .7152*this.pixels[p+1]+
 .0722*this.pixels[p+2];

 // 透明区域按白底参与能量计算，输出仍保留透明度。
 this.gray[i]=alpha*luminance+255*(1-alpha);
 }

 for(let y=0;y<this.h;y++){
 for(let x=0;x<this.w;x++){
 this.energy[y*this.stride+x]=this.gradient(x,y);
 }
 }
 }

 gradient(x,y){
 const s=this.stride,g=this.gray;
 const left=Math.max(0,x-1);
 const right=Math.min(this.w-1,x+1);
 const up=Math.max(0,y-1)*s;
 const down=Math.min(this.h-1,y+1)*s;
 const row=y*s;

 const dx=
 g[up+right]+2*g[row+right]+g[down+right]-
 g[up+left]-2*g[row+left]-g[down+left];

 const dy=
 g[down+left]+2*g[down+x]+g[down+right]-
 g[up+left]-2*g[up+x]-g[up+right];

 return Math.abs(dx)+Math.abs(dy);
 }

 findSeam(){
 const w=this.w,h=this.h,s=this.stride;
 const e=this.energy,c=this.cost,p=this.parent;

 for(let x=0;x<w;x++) c[x]=e[x];

 for(let y=1;y<h;y++){
 for(let x=0;x<w;x++){
 const at=y*s+x;
 const above=at-s;
 let best=c[above],direction=0;

 if(x>0&&c[above-1]<best){
 best=c[above-1];
 direction=-1;
 }
 if(x+1<w&&c[above+1]<best){
 best=c[above+1];
 direction=1;
 }

 c[at]=best+e[at];
 p[at]=direction;
 }
 }

 let x=0;
 const last=(h-1)*s;

 for(let i=1;i<w;i++){
 if(c[last+i]<c[last+x]) x=i;
 }

 for(let y=h-1;y>=0;y--){
 this.seam[y]=x;
 x+=p[y*s+x];
 }

 return this.seam;
 }

 removeSeam(){
 const seam=this.findSeam();
 const w=this.w,h=this.h,s=this.stride;

 for(let y=0;y<h;y++){
 const at=y*s+seam[y];
 const end=y*s+w;

 this.pixels.copyWithin(at*4,(at+1)*4,end*4);
 this.gray.copyWithin(at,at+1,end);
 this.energy.copyWithin(at,at+1,end);
 }

 this.w--;

 // 只重算 seam 附近受影响的 Sobel 邻域。
 for(let y=0;y<h;y++){
 const previous=seam[Math.max(0,y-1)];
 const current=seam[y];
 const next=seam[Math.min(h-1,y+1)];

 const lo=Math.max(
 0,Math.min(previous,current,next)-2
 );
 const hi=Math.min(
 this.w-1,Math.max(previous,current,next)+2
 );

 for(let x=lo;x<=hi;x++){
 this.energy[y*s+x]=this.gradient(x,y);
 }
 }
 }

 output(){
 const pixels=new Uint8ClampedArray(this.w*this.h*4);

 for(let y=0;y<this.h;y++){
 const start=y*this.stride*4;
 pixels.set(
 this.pixels.subarray(start,start+this.w*4),
 y*this.w*4
 );
 }

 return {width:this.w,height:this.h,pixels};
 }
 }

 function transpose(image){
 const w=image.width,h=image.height;
 const input=image.pixels;
 const output=new Uint8ClampedArray(input.length);

 for(let y=0;y<h;y++){
 for(let x=0;x<w;x++){
 const from=(y*w+x)*4;
 const to=(x*h+y)*4;
 output[to]=input[from];
 output[to+1]=input[from+1];
 output[to+2]=input[from+2];
 output[to+3]=input[from+3];
 }
 }

 return {width:h,height:w,pixels:output};
 }

 function blurSigma(image){
 // 20% 强度：σ 基准 15、随尺寸轻度放大，小图可见、大图均匀。
 return .2*Math.max(15,(image.width+image.height)/200);
 }

 // 可分离高斯模糊：预乘 alpha 后横纵两趟卷积，避免透明边缘晕色。
 function gaussianBlur(image,sigma){
 if(sigma<.5) return image;

 const w=image.width,h=image.height;
 const pixels=image.pixels;

 const radius=Math.max(1,Math.ceil(sigma*3));
 const kernel=new Float32Array(radius*2+1);
 const twoSigma2=2*sigma*sigma;

 let sum=0;
 for(let i=-radius;i<=radius;i++){
 const value=Math.exp(-(i*i)/twoSigma2);
 kernel[i+radius]=value;
 sum+=value;
 }
 for(let i=0;i<kernel.length;i++) kernel[i]/=sum;

 const src=new Float32Array(w*h*4);

 for(let i=0;i<w*h;i++){
 const p=i*4;
 const a=pixels[p+3]/255;
 src[p]=pixels[p]*a;
 src[p+1]=pixels[p+1]*a;
 src[p+2]=pixels[p+2]*a;
 src[p+3]=pixels[p+3];
 }

 const tmp=new Float32Array(w*h*4);

 for(let y=0;y<h;y++){
 const row=y*w;

 for(let x=0;x<w;x++){
 let r=0,g=0,b=0,a=0;

 for(let k=-radius;k<=radius;k++){
 const xx=Math.min(w-1,Math.max(0,x+k));
 const at=(row+xx)*4;
 const weight=kernel[k+radius];

 r+=src[at]*weight;
 g+=src[at+1]*weight;
 b+=src[at+2]*weight;
 a+=src[at+3]*weight;
 }

 const o=(row+x)*4;
 tmp[o]=r;tmp[o+1]=g;tmp[o+2]=b;tmp[o+3]=a;
 }
 }

 const out=new Uint8ClampedArray(w*h*4);

 for(let y=0;y<h;y++){
 for(let x=0;x<w;x++){
 let r=0,g=0,b=0,a=0;

 for(let k=-radius;k<=radius;k++){
 const yy=Math.min(h-1,Math.max(0,y+k));
 const at=(yy*w+x)*4;
 const weight=kernel[k+radius];

 r+=tmp[at]*weight;
 g+=tmp[at+1]*weight;
 b+=tmp[at+2]*weight;
 a+=tmp[at+3]*weight;
 }

 const o=(y*w+x)*4;
 const alpha=a/255;

 out[o]=alpha>0?r/alpha:r;
 out[o+1]=alpha>0?g/alpha:g;
 out[o+2]=alpha>0?b/alpha:b;
 out[o+3]=a;
 }
 }

 return {width:w,height:h,pixels:out};
 }

 /* ---- 主体轮廓低边形化 ---- */
 function simplifyDP(points,tolerance){
  const keep=new Uint8Array(points.length);
  keep[0]=keep[points.length-1]=1;
  const stack=[[0,points.length-1]];
  const tol2=tolerance*tolerance;

  while(stack.length){
   const [a,b]=stack.pop();
   const [ax,ay]=points[a],[bx,by]=points[b];
   const dx=bx-ax,dy=by-ay;
   const len2=dx*dx+dy*dy;
   let maxD=-1,idx=-1;

   for(let i=a+1;i<b;i++){
    const [px,py]=points[i];
    let d2;

    if(len2===0){
     const ex=px-ax,ey=py-ay;
     d2=ex*ex+ey*ey;
    }else{
     let t=((px-ax)*dx+(py-ay)*dy)/len2;
     t=Math.max(0,Math.min(1,t));
     const ex=px-(ax+t*dx),ey=py-(ay+t*dy);
     d2=ex*ex+ey*ey;
    }

    if(d2>maxD){maxD=d2;idx=i;}
   }

   if(maxD>tol2){
    keep[idx]=1;
    stack.push([a,idx],[idx,b]);
   }
  }

  return points.filter((_,i)=>keep[i]);
 }

 function fillPolygon(poly,w,h){
  const mask=new Uint8Array(w*h);

  for(let y=0;y<h;y++){
   const xs=[];

   for(let i=0;i<poly.length;i++){
    const [x1,y1]=poly[i];
    const [x2,y2]=poly[(i+1)%poly.length];

    if((y1<=y&&y2>y)||(y2<=y&&y1>y)){
     xs.push(x1+(y-y1)/(y2-y1)*(x2-x1));
    }
   }

   xs.sort((a,b)=>a-b);

   for(let i=0;i+1<xs.length;i+=2){
    const from=Math.max(0,Math.ceil(xs[i]));
    const to=Math.min(w-1,Math.floor(xs[i+1]));

    for(let x=from;x<=to;x++)mask[y*w+x]=1;
   }
  }

  return mask;
 }

 // 返回简化后的多边形顶点数；无可折角化的轮廓时返回 0。
 function polygonize(image){
  const w=image.width,h=image.height,p=image.pixels;
  const n=w*h;
  const bin=new Uint8Array(n);
  let opaque=0;

  for(let i=0;i<n;i++){
   if(p[i*4+3]>127){bin[i]=1;opaque++;}
  }

  const coverage=opaque/n;
  if(coverage>.985||coverage<.02)return 0;

  // 最大连通域（扫描序起点即上边界最左点）。
  const label=new Int32Array(n);
  const stack=new Int32Array(n);
  let bestId=0,bestSize=0,bestStart=-1,id=0;

  for(let seed=0;seed<n;seed++){
   if(!bin[seed]||label[seed])continue;
   id++;
   let sp=0,size=0;
   stack[sp++]=seed;
   label[seed]=id;

   while(sp){
    const cur=stack[--sp];
    size++;
    const x=cur%w;

    if(x>0&&bin[cur-1]&&!label[cur-1]){label[cur-1]=id;stack[sp++]=cur-1;}
    if(x+1<w&&bin[cur+1]&&!label[cur+1]){label[cur+1]=id;stack[sp++]=cur+1;}
    if(cur>=w&&bin[cur-w]&&!label[cur-w]){label[cur-w]=id;stack[sp++]=cur-w;}
    if(cur+w<n&&bin[cur+w]&&!label[cur+w]){label[cur+w]=id;stack[sp++]=cur+w;}
   }

   if(size>bestSize){bestSize=size;bestId=id;bestStart=seed;}
  }

  // Moore 8 邻域边界追踪。
  const dirs=[[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]];
  const inside=(x,y)=>
   x>=0&&y>=0&&x<w&&y<h&&label[y*w+x]===bestId;

  let px=bestStart%w;
  let py=(bestStart-px)/w;
  const startX=px,startY=py;
  const path=[];
  let bx=px-1,by=py;
  let guard=n*4+8;

  while(guard-->0){
   path.push(px,py);

   let from=dirs.findIndex(d=>d[0]===bx-px&&d[1]===by-py);
   if(from<0)from=0;
   let nextX=0,nextY=0,lastBgX=bx,lastBgY=by,found=false;

   for(let k=1;k<=8;k++){
    const d=dirs[(from+k)%8];
    const tx=px+d[0],ty=py+d[1];

    if(inside(tx,ty)){nextX=tx;nextY=ty;found=true;break;}
    lastBgX=tx;lastBgY=ty;
   }

   if(!found)break;

   px=nextX;py=nextY;bx=lastBgX;by=lastBgY;

   if(px===startX&&py===startY)break;
  }

  if(path.length<12)return 0;

  const points=[];
  for(let i=0;i<path.length;i+=2)points.push([path[i],path[i+1]]);

  // 折角强度：容差随图像尺寸放大，段数少、折角明显。
  const tolerance=Math.max(5,.015*Math.hypot(w,h));
  const simple=simplifyDP(points,tolerance);

  if(simple.length<3)return 0;

  const mask=fillPolygon(simple,w,h);

  for(let i=0;i<n;i++){
   p[i*4+3]=mask[i]?255:0;
  }

  return simple.length;
 }

 self.onmessage=({data})=>{
 try{
 const {image,targetWidth,targetHeight}=data;
 const total=
 image.width-targetWidth+
 image.height-targetHeight;

 let done=0,lastUpdate=0;

 function progress(){
 done++;
 const now=performance.now();

 if(now-lastUpdate>70||done===total){
 self.postMessage({
 type:"progress",
 value:done/Math.max(1,total)
 });
 lastUpdate=now;
 }
 }

 function shrinkWidth(input,target){
 if(input.width===target) return input;

 const carver=new Carver(input);
 while(carver.w>target){
 carver.removeSeam();
 progress();
 }
 return carver.output();
 }

 let output=shrinkWidth(image,targetWidth);

 if(output.height!==targetHeight){
 output=transpose(
 shrinkWidth(transpose(output),targetHeight)
 );
 }

 // 处理结果统一叠加 20% 高斯模糊。
 self.postMessage({type:"blur"});
 output=gaussianBlur(output,blurSigma(output));

 // 主体轮廓折角化（低边形外轮廓）。
 let vertices=0;

 if(data.polygonize){
  self.postMessage({type:"poly"});
  vertices=polygonize(output);
 }

 self.postMessage(
 {type:"done",image:output,polygonized:vertices},
 [output.pixels.buffer]
 );
 }catch(error){
 self.postMessage({
 type:"error",
 message:error.message
 });
 }
 };
}

const $=id=>document.getElementById(id);
const colors=["#1FC3FC","#FFD731","#61E082"];
let colorIndex=0;
let original=null;
let worker=null;
let busy=false;
let resultReady=false;

const source=$("source");
const frame=$("frame");
const result=$("result");

// 白色抖动描边：跟随图片实际边缘（框内居中位置），粗细与抖动随尺寸自适应。
function updateOutline(){
 const w=source.offsetWidth,h=source.offsetHeight;

 if(!w||!h)return;

 const stroke=Math.max(3,Math.min(8,Math.min(w,h)*.012));
 const r=$("outlineRect");

 r.setAttribute("x",source.offsetLeft+1);
 r.setAttribute("y",source.offsetTop+1);
 r.setAttribute("width",w-2);
 r.setAttribute("height",h-2);
 r.setAttribute("stroke-width",stroke);

 const map=document.querySelector("#wobble feDisplacementMap");
 map.setAttribute("scale",Math.max(2,Math.min(6,Math.min(w,h)*.008)));
}

new ResizeObserver(updateOutline).observe(frame);
const workerURL=URL.createObjectURL(
 new Blob(
 ["("+seamWorker.toString()+")()"],
 {type:"text/javascript"}
 )
);

// 主体识别：U2Netp 显著主体检测（本地 ONNX WASM 推理，任意主体通用）。
let sessionPromise=null;

function getSession(){
 if(!sessionPromise){
 sessionPromise=(async()=>{
 // 动态加载 ORT 运行时（UMD，全局 ort）。
 await new Promise((resolve,reject)=>{
 const script=document.createElement("script");
 script.src="ai/ort/ort.min.js";
 script.onload=resolve;
 script.onerror=()=>reject(Error("ORT 脚本加载失败"));
 document.head.appendChild(script);
 });

 ort.env.wasm.wasmPaths="ai/ort/";
 ort.env.wasm.numThreads=1;

 const response=await fetch("ai/u2netp.onnx");
 const buffer=await response.arrayBuffer();

 return ort.InferenceSession.create(buffer,{
 executionProviders:["wasm"],
 graphOptimizationLevel:"all"
 });
 })();
 }
 return sessionPromise;
}

// 返回主体抠图（背景 alpha=0），没有明显主体时返回 null。
async function extractSubject(){
 const session=await getSession();
 const resolution=320;

 const small=document.createElement("canvas");
 small.width=resolution;
 small.height=resolution;
 const sctx=small.getContext("2d");
 sctx.drawImage(source,0,0,resolution,resolution);

 const data=sctx.getImageData(0,0,resolution,resolution).data;
 const stride=resolution*resolution;
 const input=new Float32Array(stride*3);

 // 模型导出约定：RGB、(v-128)/256、CHW。
 for(let i=0,j=0;i<data.length;i+=4,j++){
 input[j]=(data[i]-128)/256;
 input[stride+j]=(data[i+1]-128)/256;
 input[stride*2+j]=(data[i+2]-128)/256;
 }

 const tensor=new ort.Tensor("float32",input,[1,3,resolution,resolution]);
 const outputs=await session.run({[session.inputNames[0]]:tensor});
 const saliency=outputs[session.outputNames[0]].data;

 // 显著图 → alpha：压掉背景残留，保留柔边。
 const mask=document.createElement("canvas");
 mask.width=resolution;
 mask.height=resolution;
 const mctx=mask.getContext("2d");
 const image=mctx.createImageData(resolution,resolution);
 let peak=0,total=0;

 for(let i=0;i<stride;i++){
 const raw=Math.min(1,Math.max(0,Number(saliency[i])));
 peak=Math.max(peak,raw);

 let value=(raw-.3)/.4;
 value=Math.min(1,Math.max(0,value));
 value=value*value*(3-2*value);
 total+=value;

 const channel=Math.round(value*255);
 image.data[i*4]=channel;
 image.data[i*4+1]=channel;
 image.data[i*4+2]=channel;
 image.data[i*4+3]=255;
 }

 if(peak<.35||total/stride<.02) return null;

 mctx.putImageData(image,0,0);

 const big=document.createElement("canvas");
 big.width=original.width;
 big.height=original.height;
 const bctx=big.getContext("2d");
 bctx.imageSmoothingEnabled=true;
 bctx.imageSmoothingQuality="high";
 bctx.drawImage(mask,0,0,big.width,big.height);

 const alpha=bctx.getImageData(0,0,big.width,big.height).data;
 const pixels=new Uint8ClampedArray(original.pixels);

 for(let i=0;i<pixels.length;i+=4){
 pixels[i+3]=alpha[i+1];
 }

 return {pixels};
}

function showResult(show){
 $("originalView").hidden=show;
 $("resultView").hidden=!show;
 $("controls").hidden=show;
 $("actions").hidden=!show;
}

function setBusy(value){
 busy=value;
 $("upload").disabled=value;
 $("horizontal").disabled=value;
 $("vertical").disabled=value;
}

function finish(message=""){
 worker?.terminate();
 worker=null;
 setBusy(false);
 setStartLabel(false);
 $("status").textContent=message;
}

function setStartLabel(busy){
 const start=$("start");
 const img=start.querySelector("img");
 const span=start.querySelector("span");

 if(img)img.hidden=busy;
 if(span)span.hidden=!busy;
}

$("upload").onchange=async event=>{
 const file=event.target.files[0];
 if(!file||busy) return;

 setBusy(true);
 let bitmap;

 try{
 if(!["image/jpeg","image/png","image/webp"].includes(file.type)){
 throw Error("请选择 JPG、PNG 或 WEBP 图片。");
 }

 bitmap=await createImageBitmap(file);

 if(
 bitmap.width*bitmap.height>4000000||
 bitmap.width>4096||
 bitmap.height>4096
 ){
 throw Error("请选择不超过 400 万像素、单边不超过 4096 的图片。");
 }

 source.width=bitmap.width;
 source.height=bitmap.height;

 const context=source.getContext("2d");
 context.drawImage(bitmap,0,0);

 original={
 width:bitmap.width,
 height:bitmap.height,
 pixels:context.getImageData(
 0,0,bitmap.width,bitmap.height
 ).data
 };

 source.hidden=false;
 frame.classList.add("show");
 requestAnimationFrame(updateOutline);
 $("status").textContent="";
 resultReady=false;
 showResult(false);
 }catch(error){
 $("status").textContent=error.message;
 }finally{
 bitmap?.close();
 setBusy(false);
 }
};

$("start").onclick=async()=>{
 if(busy) return;

 if(!original){
 $("status").textContent="请先上传一张图片。";
 return;
 }

 // 滑块从不压缩到删除该方向 85% 的像素尺寸。
 const horizontal=Number($("horizontal").value)/100;
 const vertical=Number($("vertical").value)/100;

 const targetWidth=Math.max(
 1,Math.round(original.width*(1-horizontal*.85))
 );
 const targetHeight=Math.max(
 1,Math.round(original.height*(1-vertical*.85))
 );

 if(
 targetWidth===original.width&&
 targetHeight===original.height
 ){
 $("status").textContent="请调高横向或纵向滑块。";
 return;
 }

 setBusy(true);
 setStartLabel(true);

 // 第一步：识别主体物并输出透明背景。
 let pixels=new Uint8ClampedArray(original.pixels);
 let subjectExtracted=false;

 try{
 const subject=await extractSubject();

 if(subject){
 pixels=subject.pixels;
 subjectExtracted=true;
 }
 }catch(error){
 console.warn(error);
 }

 // 第二步：对透明背景图做内容感知缩放 + 20% 高斯模糊。
 try{
 worker=new Worker(workerURL);

 worker.onerror=()=>{
 finish("处理失败，或当前环境不支持 Web Worker。");
 };

 worker.onmessage=({data})=>{
 if(data.type==="error"){
 finish(data.message);
 }

 if(data.type==="done"){
 const image=data.image;
 result.width=image.width;
 result.height=image.height;

 result.getContext("2d").putImageData(
 new ImageData(
 image.pixels,image.width,image.height
 ),0,0
 );

 resultReady=true;
 finish();
 showResult(true);
 }
 };

 // pixels 已是独立副本（主体抠图或原图），直接转移给 worker。
 worker.postMessage({
 image:{
 width:original.width,
 height:original.height,
 pixels
 },
 targetWidth,
 targetHeight,
 polygonize:subjectExtracted
 },[pixels.buffer]);
 }catch(error){
 finish("无法启动处理线程："+error.message);
 }
};

$("back").onclick=()=>{
 showResult(false);
 $("status").textContent="";
};

$("save").onclick=()=>{
 if(!resultReady) return;

 result.toBlob(blob=>{
 if(!blob){
 $("status").textContent="保存失败，请重试。";
 return;
 }

 const url=URL.createObjectURL(blob);
 const link=document.createElement("a");
 link.href=url;
 link.download="seam-result.png";
 link.click();

 setTimeout(()=>URL.revokeObjectURL(url),5000);
 },"image/png");
};

document.addEventListener("click",event=>{
 // 操作图片、按钮、上传和滑块时不切换背景。
 if(event.target.closest("button,label,input,canvas,a")){
 return;
 }
 if(window.getSelection()?.toString()) return;

 colorIndex=(colorIndex+1)%colors.length;
 document.documentElement.style.setProperty(
 "--bg",colors[colorIndex]
 );
});

window.addEventListener("pagehide",()=>{
 worker?.terminate();
});
