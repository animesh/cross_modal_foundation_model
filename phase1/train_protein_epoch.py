"""One-epoch protein continued training, independent small-corpus experiment.
Reuses the locally tested SDPA model without changing it. No RNA control or
single-cell benchmark evaluation is performed by this script.
"""
from pathlib import Path
import json,csv,hashlib,shutil
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import load_file,save_file
import test_tahoe_original as adapter

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/protein_continued_seed42'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def sample_profile(ids,values,cls,rng):
    measured=np.flatnonzero(values>0)
    if len(measured)<2:raise ValueError('Fewer than two measured genes')
    # Quantiles of all measured proteins before sequence subsampling.
    v=torch.tensor(values[measured],dtype=torch.float32)
    boundaries=torch.quantile(v,torch.linspace(0,1,50))
    bins=(torch.bucketize(v,boundaries,right=False)+1).numpy()
    chosen=np.sort(rng.choice(len(measured),min(1023,len(measured)),replace=False))
    genes=np.r_[cls,ids[measured[chosen]]].astype(np.int64)
    targets=np.r_[0,bins[chosen]].astype(np.float32)
    masked=rng.random(len(genes))<0.5;masked[0]=False
    if not masked.any():masked[1]=True
    return genes,targets,masked

def run():
    assert torch.cuda.is_available(),'CUDA unavailable'
    assert not OUT.exists(),'Output already exists; refusing to overwrite training run'
    source=ROOT/'results/protein_training_inputs/profiles.npz'
    with np.load(source,allow_pickle=False) as z:
        ids=z['gene_ids'];values=z['lfq'];samples=z['samples']
    assert values.shape==(308,5763) and np.isfinite(values).all() and (values>=0).all()
    assert len(set(ids))==len(ids) and ids.min()>0 and ids.max()<62720
    vocab=json.loads((adapter.MODEL/'vocab.json').read_text())
    weights=adapter.MODEL/'model.safetensors'
    assert sha(weights)==adapter.EXPECTED_SHA,'Starting checkpoint hash mismatch'
    model=adapter.Model();state=load_file(str(weights),device='cpu')
    model.load_state_dict(state,strict=True);del state
    model.cuda().train()
    torch.manual_seed(42);rng=np.random.default_rng(42)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.01,foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=1.0)
    OUT.mkdir(parents=True)
    manifest={'scope':'Independent one-study protein continued-training experiment, NOT full paper reproduction','starting_checkpoint_sha256':sha(weights),'profiles_sha256':sha(source),'adapter_sha256':sha(Path(adapter.__file__)),'seed':42,'epochs':1,'batch_size':1,'max_length':1024,'learning_rate':1e-5,'weight_decay':0.01,'mask_probability':0.5,'precision':'FP32 weights, FP16 autocast','attention':'PyTorch efficient SDPA','preprocessing':'Positive measured LFQ only; 51-bin per-profile quantiles before uniform subsampling; no missing targets or imputation','loss':'Mean of masked expression-decoder and MVC MSE','limits':'No RNA-only continued-training control; no single-cell evaluation; no original FlashAttention equivalence. Training dataset cannot be used as held-out protein evaluation.','status':'RUNNING'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    initial=model.model.expression_decoder.out_proj.weight.detach().clone()
    order=rng.permutation(len(samples))
    with (OUT/'training.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step','sample','tokens','masked','loss','attempts','scale','peak_GiB']);writer.writeheader()
        for step,index in enumerate(order,1):
            genes,target,mask=sample_profile(ids,values[index],vocab['<cls>'],rng)
            genes=torch.tensor(genes[None],device='cuda');target=torch.tensor(target[None],device='cuda');mask=torch.tensor(mask[None],device='cuda')
            expr=target.masked_fill(mask,-1)
            updated=False
            for attempt in range(1,13):
                optimizer.zero_grad(set_to_none=True)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),torch.autocast('cuda',dtype=torch.float16):
                    outputs=model(genes,expr,mask)
                    loss=(F.mse_loss(outputs['expr_preds'][mask].float(),target[mask])+F.mse_loss(outputs['mvc_output'][mask].float(),target[mask]))/2
                assert torch.isfinite(loss),'Nonfinite loss at sample '+str(samples[index])
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                assert all(p.grad is not None for p in model.parameters()),'Missing gradients'
                finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                scaler.step(optimizer);scaler.update()
                if finite:updated=True;break
            assert updated,'Failed to obtain finite update: '+str(samples[index])
            record=dict(step=step,sample=str(samples[index]),tokens=genes.shape[1],masked=int(mask.sum()),loss=float(loss.detach()),attempts=attempt,scale=scaler.get_scale(),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
            writer.writerow(record);stream.flush()
            if step==1 or step%25==0 or step==len(samples):print(json.dumps(record),flush=True)
    assert not torch.equal(initial,model.model.expression_decoder.out_proj.weight),'No learned weight change'
    assert len(optimizer.state)==len(list(model.parameters())),'Incomplete Adam states'
    optimizer.zero_grad(set_to_none=True)
    final={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in final.values()),'Nonfinite final weights'
    save_file(final,str(OUT/'model.safetensors'))
    reloaded=load_file(str(OUT/'model.safetensors'),device='cpu')
    assert set(final)==set(reloaded) and all(torch.equal(final[k],reloaded[k]) for k in final),'Checkpoint roundtrip failed'
    for name in ['vocab.json','model_config.yml','collator_config.yml']:shutil.copyfile(adapter.MODEL/name,OUT/name)
    manifest.update(status='COMPLETE',successful_updates=len(samples),trained_checkpoint_sha256=sha(OUT/'model.safetensors'))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Trained checkpoint:',(OUT/'model.safetensors').resolve())

run()
