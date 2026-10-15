import torch
from torch import nn
from .layers import Encoder,boundary,region,image_demo


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',24)
        self.encoder=Encoder(3,w);self.region=nn.Conv2d(w,1,1);self.boundary=nn.Conv2d(w,1,1);self.temperature=nn.Conv2d(w,1,1)
    def predict(self,x):
        h=self.encoder(x);boundary=self.boundary(h).sigmoid();temperature=1+1.5*boundary*self.temperature(h).sigmoid()
        if self.c['variant'] in {'baseline','no_temperature'}:temperature=torch.ones_like(temperature)
        return self.region(h).div(temperature).sigmoid(),boundary,temperature
    def forward(self,b):
        p,edge,t=self.predict(b['x'].float())
        view=(b['x'].float()*1.1+.03).clamp(0,1)
        p2,edge2,_=self.predict(view)
        return {'probability':p,'boundary':edge,'temperature':t,'view_probability':p2,'view_boundary':edge2}
    def objective(self,b,p):
        y=b['y'].float();q=boundary(y,3)
        loss=(region(p['probability'],y)+region(p['view_probability'],y))/2
        if self.c['variant']=='baseline':return loss
        loss=loss+.3*region(p['boundary'],q)
        if self.c['variant']!='no_calibration':loss=loss+.1*((1+2*q)*(p['probability']-y).square()).sum()/(1+2*q).sum()
        if self.c['variant']!='no_consistency':
            a=p['probability'].clamp(1e-6,1-1e-6);v=p['view_probability'].clamp(1e-6,1-1e-6)
            entropy=lambda z:-(z*z.log()+(1-z)*(1-z).log())/.693147
            weight=(1-(entropy(a)+entropy(v))/2).detach()
            loss=loss+.2*(weight*(abs(a-v)+.5*abs(p['boundary']-p['view_boundary']))).sum()/weight.sum().clamp_min(1)
        return loss


def demo(seed,n,c):return image_demo(seed,n)
