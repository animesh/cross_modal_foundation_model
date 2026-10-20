"""GPU feasibility diagnostic ONLY: random 70M proxy, not Tahoe-x1.
Requires only the existing torch installation. No downloads or real-data training.
"""
import gc, json, time, traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint

OUT = Path(__file__).resolve().parent / 'results/gpu_70m_test.json'

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1=nn.LayerNorm(512); self.n2=nn.LayerNorm(512)
        self.qkv=nn.Linear(512,1536); self.proj=nn.Linear(512,512)
        self.ff=nn.Sequential(nn.Linear(512,2048),nn.GELU(),nn.Linear(2048,512))
    def forward(self,x):
        b,l,_=x.shape
        q,k,v=self.qkv(self.n1(x)).reshape(b,l,3,8,64).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
        x=x+self.proj(a.transpose(1,2).reshape(b,l,512))
        return x+self.ff(self.n2(x))

class Proxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(62000,512)
        self.blocks=nn.ModuleList([Block() for _ in range(12)])
        self.head=nn.Linear(512,1)
        current=sum(p.numel() for p in self.parameters())
        self.extra=nn.Parameter(torch.zeros(70000000-current))
    def forward(self,ids):
        x=self.emb(ids)
        for block in self.blocks:
            x=checkpoint(block,x,use_reentrant=False)
        # Include extra parameters in gradient and Adam-state allocation.
        return self.head(x).squeeze(-1)+self.extra.mean()

def run():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in this Python environment')
    torch.manual_seed(42)
    free,total=torch.cuda.mem_get_info()
    report=dict(scope='Synthetic hardware feasibility proxy, NOT original Tahoe checkpoint or biological reproduction',torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),free_GiB=free/2**30,total_GiB=total/2**30,parameters=70000000,architecture='12 blocks, width512, 8 heads, FF2048, vocabulary62000; extra trainable vector to reach exactly70M',checkpointing=True,tests=[])
    print('GPU:',report['gpu'],'free GiB:',round(report['free_GiB'],2),flush=True)
    # Test efficient attention alone first; unsupported kernels do not prevent math tests.
    for name,backend in [('efficient',SDPBackend.EFFICIENT_ATTENTION),('math',SDPBackend.MATH)]:
        try:
            q=torch.randn(1,8,128,64,device='cuda',dtype=torch.float16,requires_grad=True)
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q,q,q).float().square().mean().backward()
            torch.cuda.synchronize()
            assert torch.isfinite(q.grad).all()
            report['tests'].append(dict(kind='attention',backend=name,status='PASS'))
        except RuntimeError as exc:
            report['tests'].append(dict(kind='attention',backend=name,status='FAILED',error=str(exc)))
        finally:
            q=None;gc.collect();torch.cuda.empty_cache()
    efficient=any(r.get('backend')=='efficient' and r['status']=='PASS' for r in report['tests'])
    backend=SDPBackend.EFFICIENT_ATTENTION if efficient else SDPBackend.MATH
    for length in [256,512,1024,2048]:
        model=optimizer=ids=target=loss=scaler=None
        row=dict(kind='70M_forward_backward_Adam',batch=1,sequence_length=length,backend='efficient' if efficient else 'math',precision='FP32 parameters, FP16 autocast',steps=2)
        try:
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            model=Proxy().cuda()
            assert sum(p.numel() for p in model.parameters())==70000000
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
            scaler=torch.amp.GradScaler('cuda',init_scale=128.0)
            ids=torch.randint(0,62000,(1,length),device='cuda')
            target=torch.randn(1,length,device='cuda')
            before=model.head.weight.detach().clone()
            successful_steps=0
            attempts=0
            skipped_scales=[]
            while successful_steps<2 and attempts<12:
                attempts+=1
                optimizer.zero_grad(set_to_none=True)
                with sdpa_kernel(backend),torch.autocast('cuda',dtype=torch.float16):
                    pred=model(ids)
                    loss=F.mse_loss(pred.float(),target)
                assert torch.isfinite(loss), "Nonfinite forward loss"
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                missing=[name for name,p in model.named_parameters() if p.grad is None]
                assert not missing, "Missing gradients: "+repr(missing)
                invalid=[name for name,p in model.named_parameters() if not torch.isfinite(p.grad).all()]
                scale_before=float(scaler.get_scale())
                scaler.step(optimizer);scaler.update()
                if invalid:
                    skipped_scales.append(dict(scale=scale_before,next_scale=float(scaler.get_scale()),parameters=invalid))
                    print('Skipped overflow update; reducing scale:',scale_before,'->',scaler.get_scale(),flush=True)
                    continue
                successful_steps+=1
            assert successful_steps==2, "Could not obtain two finite-gradient Adam updates: "+repr(skipped_scales)
            torch.cuda.synchronize()
            assert not torch.equal(before,model.head.weight), "Head weights did not update"
            assert len(optimizer.state)==len(list(model.parameters())), "Incomplete Adam state allocation"
            row.update(status='PASS',successful_steps=successful_steps,attempts=attempts,skipped_overflows=skipped_scales,loss=float(loss.detach()),seconds=time.perf_counter()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        except (RuntimeError,AssertionError) as exc:
            row.update(status='FAILED',error=str(exc) or repr(exc),exception_type=type(exc).__name__,traceback=traceback.format_exc(),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        finally:
            model=optimizer=ids=target=loss=scaler=None
            if 'pred' in locals(): pred=None
            if 'before' in locals(): before=None
            gc.collect();torch.cuda.empty_cache()
        report['tests'].append(row)
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
        if row['status']=='FAILED': break
    print('Report:',OUT.resolve())

if __name__=='__main__':
    run()
