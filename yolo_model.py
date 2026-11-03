"""Model, loss and postprocessing from Person 3’s 2026-11-03 baseline archive (core definitions verified identical)."""

import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CLASSES = ('drill', 'scissor', 'screwdriver')

SEED = 288275

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def xywh_to_xyxy(x):
    return torch.cat((x[..., :2]-x[..., 2:]/2, x[..., :2]+x[..., 2:]/2), -1)

def box_iou(a,b):
    inter=(torch.minimum(a[:,None,2:],b[None,:,2:])-torch.maximum(a[:,None,:2],b[None,:,:2])).clamp(min=0).prod(-1)
    area_a=(a[:,2:]-a[:,:2]).clamp(min=0).prod(-1)
    area_b=(b[:,2:]-b[:,:2]).clamp(min=0).prod(-1)
    return inter/(area_a[:,None]+area_b[None,:]-inter+1e-9)

def wh_iou(a,b):
    inter=torch.minimum(a[:,None],b[None]).prod(-1)
    return inter/(a.prod(-1)[:,None]+b.prod(-1)[None]-inter+1e-9)

def cluster_anchors(wh,k=5,seed=SEED):
    if len(wh)<k:raise ValueError('Not enough training boxes to cluster anchors')
    generator=torch.Generator().manual_seed(seed);centers=wh[torch.randperm(len(wh),generator=generator)[:k]].clone()
    for _ in range(100):
        assign=wh_iou(wh,centers).argmax(1)
        updated=torch.stack([wh[assign==j].median(0).values if (assign==j).any() else centers[j] for j in range(k)])
        if torch.allclose(updated,centers,atol=1e-6):break
        centers=updated
    return centers[centers.prod(1).argsort()].clamp(min=1/640)

def conv_bn(cin,cout,k=3,stride=1):
    return nn.Sequential(nn.Conv2d(cin,cout,k,stride,k//2,bias=False),nn.BatchNorm2d(cout),nn.LeakyReLU(.1,inplace=True))

class BasicBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__();self.c1=nn.Conv2d(cin,cout,3,stride,1,bias=False);self.b1=nn.BatchNorm2d(cout)
        self.c2=nn.Conv2d(cout,cout,3,1,1,bias=False);self.b2=nn.BatchNorm2d(cout)
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x):return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))+self.skip(x))

class YOLOv2ResNet18(nn.Module):
    def __init__(self,anchors,classes=3):
        super().__init__();self.register_buffer('anchors',anchors.clone().float());self.classes=classes
        self.stem=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(3,2,1))
        self.s1=nn.Sequential(BasicBlock(64,64),BasicBlock(64,64))
        self.s2=nn.Sequential(BasicBlock(64,128,2),BasicBlock(128,128))
        self.s3=nn.Sequential(BasicBlock(128,256,2),BasicBlock(256,256))
        self.s4=nn.Sequential(BasicBlock(256,512,2),BasicBlock(512,512))
        self.deep=nn.Sequential(conv_bn(512,1024),conv_bn(1024,1024))
        self.passthrough=conv_bn(256,64,1)
        self.head=nn.Sequential(conv_bn(1280,1024),nn.Conv2d(1024,len(anchors)*(5+classes),1))
    def forward(self,x):
        fine=self.s3(self.s2(self.s1(self.stem(x))))
        deep=self.deep(self.s4(fine));route=F.pixel_unshuffle(self.passthrough(fine),2)
        raw=self.head(torch.cat((deep,route),1));b,_,h,w=raw.shape
        return raw.view(b,len(self.anchors),5+self.classes,h,w).permute(0,1,3,4,2).contiguous()

def decode(raw,anchors):
    _,a,h,w,_=raw.shape
    gy,gx=torch.meshgrid(torch.arange(h,device=raw.device),torch.arange(w,device=raw.device),indexing='ij')
    grid=torch.stack((gx,gy),-1).to(raw.dtype)
    xy=(raw[...,:2].sigmoid()+grid)/raw.new_tensor([w,h])
    wh=raw[...,2:4].clamp(-8,8).exp()*anchors[None,:,None,None,:]
    return torch.cat((xy,wh),-1)

def yolo_loss(raw,targets,anchors,ignore_iou=.6):
    b,a,h,w,_=raw.shape;pos=torch.zeros(raw.shape[:-1],device=raw.device,dtype=torch.bool)
    target=torch.zeros_like(raw);assigned_boxes=torch.zeros_like(raw[...,:4]);collisions=0
    for bi,gt in enumerate(targets):
        for row in gt[torch.argsort(gt[:,3]*gt[:,4],descending=True)]:
            cls,x,y,bw,bh=row.tolist();ix=min(int(x*w),w-1);iy=min(int(y*h),h-1)
            ranking=wh_iou(row[None,3:5].to(raw.device),anchors).argsort(1,descending=True)[0]
            slot=next((int(j) for j in ranking if not pos[bi,j,iy,ix]),None)
            if slot is None:collisions+=1;continue
            pos[bi,slot,iy,ix]=True
            target[bi,slot,iy,ix,:2]=raw.new_tensor([x*w-ix,y*h-iy])
            target[bi,slot,iy,ix,2:4]=torch.log(row[3:5].to(raw.device)/anchors[slot]+1e-9)
            target[bi,slot,iy,ix,5]=cls
            assigned_boxes[bi,slot,iy,ix]=row[1:5].to(raw.device)
    predicted=decode(raw,anchors);negative=~pos
    with torch.no_grad():
        for bi,gt in enumerate(targets):
            if len(gt):
                overlap=box_iou(xywh_to_xyxy(predicted[bi].reshape(-1,4)),xywh_to_xyxy(gt[:,1:5].to(raw.device))).max(1).values
                negative[bi]&=overlap.reshape(a,h,w)<ignore_iou
    zero=raw.sum()*0
    if pos.any():
        loc=F.binary_cross_entropy_with_logits(raw[...,:2][pos],target[...,:2][pos],reduction='sum')+F.mse_loss(raw[...,2:4][pos],target[...,2:4][pos],reduction='sum')
        pp=xywh_to_xyxy(predicted[pos]);tt=xywh_to_xyxy(assigned_boxes[pos])
        inter=(torch.minimum(pp[:,2:],tt[:,2:])-torch.maximum(pp[:,:2],tt[:,:2])).clamp(min=0).prod(1)
        iou=(inter/((pp[:,2:]-pp[:,:2]).prod(1)+(tt[:,2:]-tt[:,:2]).prod(1)-inter+1e-9)).detach()
        obj=F.mse_loss(raw[...,4][pos].sigmoid(),iou,reduction='sum')
        cls=F.cross_entropy(raw[...,5:][pos],target[...,5][pos].long(),reduction='sum')
    else:loc=obj=cls=zero
    noobj=raw[...,4][negative].sigmoid().square().sum()
    loss=(5*loc+obj+.5*noobj+cls)/b
    return loss,{'loss':float(loss.detach()),'assigned':int(pos.sum()),'collisions':collisions}

def nms(boxes,scores,threshold=.5):
    order=scores.argsort(descending=True);keep=[]
    while len(order):
        i=order[0];keep.append(i)
        if len(order)==1:break
        rest=order[1:];order=rest[box_iou(boxes[i:i+1],boxes[rest])[0]<=threshold]
    return torch.stack(keep) if keep else torch.empty(0,dtype=torch.long,device=boxes.device)

def postprocess(raw,anchors,conf=.05,nms_iou=.5,topk=300):
    boxes=xywh_to_xyxy(decode(raw,anchors)).clamp(0,1);results=[]
    probs=raw[...,4:5].sigmoid()*raw[...,5:].softmax(-1)
    for b in range(len(raw)):
        flat=boxes[b].reshape(-1,4);scores,classes=probs[b].reshape(-1,len(CLASSES)).max(1)
        ok=(scores>=conf)&((flat[:,2:]-flat[:,:2]).min(1).values>0)
        flat,scores,classes=flat[ok],scores[ok],classes[ok]
        idx=scores.argsort(descending=True)[:1000];flat,scores,classes=flat[idx],scores[idx],classes[idx]
        chosen=[]
        for c in range(len(CLASSES)):
            idx=torch.where(classes==c)[0];chosen.extend(idx[nms(flat[idx],scores[idx],nms_iou)].tolist())
        idx=torch.tensor(chosen,device=raw.device,dtype=torch.long);idx=idx[scores[idx].argsort(descending=True)[:topk]]
        results.append({'boxes':flat[idx].detach().cpu(),'scores':scores[idx].detach().cpu(),'labels':classes[idx].detach().cpu()})
    return results
