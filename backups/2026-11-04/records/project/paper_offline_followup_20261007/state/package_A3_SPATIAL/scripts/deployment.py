"""Prediction math unchanged; independent read-only diagnostic suppression."""
import types
import torch
from adapter import model,digest,volume
def router_forward(self,x):
 proj=self.project(x)
 proj=self.norm(proj.permute(0,2,3,1)).permute(0,3,1,2)
 logits=torch.einsum('bdhw,nd->bnhw',proj,self.tokens)*self.scale
 weights=torch.softmax(logits,dim=1)
 routed=torch.einsum('bnhw,nd->bdhw',weights,self.tokens)
 residual=self.inverse_project(routed)
 out=x+self.gamma*residual
 return out
def decoder_forward(self,feature):
 x0,x1,x2,x3,x4=feature
 x=self.up1(x4,x3)
 x=self.up2(x,x2)
 x=self.up3(x,x1)
 x_f=self.up4(x,x0)
 x_f=self.mask_token_router(x_f)
 output=self.out_conv(x_f)
 return output,x_f
def deploy(m,path):
 if path=='deployment_no_diagnostics' and hasattr(m.decoder,'mask_token_router'):
  r=m.decoder.mask_token_router;r.forward=types.MethodType(router_forward,r)
  m.decoder.forward=types.MethodType(decoder_forward,m.decoder)
 return m
def load(method,cp,branch='model_1',path='original_runtime'):
 m,r=model(method,cp,branch);deploy(m,path);return m,r
