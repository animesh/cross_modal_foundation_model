"""Paired RNA reconstruction evaluation of original/protein/RNA checkpoints.
Not a downstream cell-representation benchmark or full paper reproduction.
No training. Identical evaluation cells, genes and masks for all checkpoints.
"""
from pathlib import Path
import json,csv,gzip,shutil,hashlib
import numpy as np
import h5py
from scipy import sparse
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import load_file
import test_tahoe_original as adapter
from prepare_rna_control import text
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/paired_rna_evaluation'

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
    assert not OUT.exists(),'Evaluation output exists; refusing overwrite'
    inputs=ROOT/'results/rna_control_inputs'
    manifest=json.loads((inputs/'manifest.json').read_text())
    source=Path(manifest['source'])
    assert source.stat().st_size==manifest['source_bytes'] and source.stat().st_mtime_ns==manifest['source_mtime_ns'],'RNA source changed'
    with np.load(inputs/'cell_split.npz',allow_pickle=False) as z:
        rows=z['evaluation_rows'];train=z['train_rows']
    assert len(rows)==1024 and len(set(rows))==1024 and not set(rows)&set(train)
    vocab=json.loads((adapter.MODEL/'vocab.json').read_text())
    checkpoints={'original':adapter.MODEL/'model.safetensors','protein':ROOT/'results/protein_continued_seed42/model.safetensors','rna_control':ROOT/'results/rna_continued_seed42/model.safetensors'}
    hashes={k:sha(p) for k,p in checkpoints.items()}
    assert hashes['original']==adapter.EXPECTED_SHA
    for name,folder in [('protein','protein_continued_seed42'),('rna_control','rna_continued_seed42')]:
        m=json.loads((ROOT/'results'/folder/'manifest.json').read_text())
        assert m['status']=='COMPLETE' and m['successful_updates']==308
        assert m['trained_checkpoint_sha256']==hashes[name] and m['starting_checkpoint_sha256']==hashes['original']
    OUT.mkdir(parents=True)
    actual=source;temporary=None
    with source.open('rb') as f:magic=f.read(2)
    if magic==bytes([31,139]):
        temporary=OUT/'source_uncompressed.h5ad'
        with gzip.open(source,'rb') as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst,8388608)
        actual=temporary
    genes_list=[];targets_list=[];masks_list=[]
    with h5py.File(actual,'r') as f:
        features=text(f['var/feature_id']);X=f['X']
        mapped=[(i,int(vocab[g.split('.')[0]])) for i,g in enumerate(features) if g.split('.')[0] in vocab and int(vocab[g.split('.')[0]])>0]
        ids=np.array(sorted({t for _,t in mapped}));lookup={int(t):i for i,t in enumerate(ids)}
        shape=X.shape if isinstance(X,h5py.Dataset) else tuple(X.attrs['shape'])
        matrix=sparse.csr_matrix((np.ones(len(mapped)),([i for i,t in mapped],[lookup[t] for i,t in mapped])),shape=(shape[1],len(ids)))
        rng=np.random.default_rng(73021)
        for index in rows:
            index=int(index)
            if isinstance(X,h5py.Dataset):row=sparse.csr_matrix(X[index:index+1])
            else:
                encoding=X.attrs['encoding-type'];encoding=encoding.decode() if isinstance(encoding,bytes) else encoding
                assert encoding=='csr_matrix'
                lo,hi=map(int,X['indptr'][index:index+2])
                row=sparse.csr_matrix((X['data'][lo:hi],X['indices'][lo:hi],np.array([0,hi-lo])),shape=(1,shape[1]))
            values=(row@matrix).toarray().ravel().astype(np.float32)
            assert np.isfinite(values).all() and (values>=0).all()
            genes,target,mask=sample_profile(ids,values,vocab['<cls>'],rng)
            assert len(genes)==1024 and mask.any() and not mask[0]
            genes_list.append(genes);targets_list.append(target);masks_list.append(mask)
    if temporary:temporary.unlink()
    genes=np.stack(genes_list);targets=np.stack(targets_list);masks=np.stack(masks_list)
    np.savez_compressed(OUT/'evaluation_inputs.npz',rows=rows,genes=genes,targets=targets,masks=masks)
    scores={};summaries=[]
    for name,path in checkpoints.items():
        model=adapter.Model();model.load_state_dict(load_file(str(path),device='cpu'),strict=True);model.cuda().eval()
        results=[];embeddings=[]
        with torch.no_grad():
            for i,index in enumerate(rows):
                g=torch.tensor(genes[i:i+1],device='cuda');t=torch.tensor(targets[i:i+1],device='cuda');mask=torch.tensor(masks[i:i+1],device='cuda')
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),torch.autocast('cuda',dtype=torch.float16):
                    output=model(g,t.masked_fill(mask,-1),mask)
                a=F.mse_loss(output['expr_preds'][mask].float(),t[mask]);b=F.mse_loss(output['mvc_output'][mask].float(),t[mask])
                assert torch.isfinite(a) and torch.isfinite(b) and torch.isfinite(output['cell_emb']).all()
                results.append({'row':int(index),'expression_mse':float(a),'mvc_mse':float(b),'mean_mse':float((a+b)/2)})
                # Masked-input CLS embeddings are diagnostic only, not standard downstream embeddings.
                embeddings.append(output['cell_emb'].float().cpu().numpy()[0])
                if (i+1)%256==0:print(name,i+1,'/1024',flush=True)
        with (OUT/(name+'_per_cell.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        np.savez_compressed(OUT/(name+'_masked_cls.npz'),rows=rows,embeddings=np.stack(embeddings))
        scores[name]=np.array([r['mean_mse'] for r in results])
        summaries.append({'model':name,'cells':len(rows),'mean_expression_mse':float(np.mean([r['expression_mse'] for r in results])),'mean_mvc_mse':float(np.mean([r['mvc_mse'] for r in results])),'mean_mse':float(scores[name].mean())})
        del model,output;torch.cuda.empty_cache()
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    differences={}
    for comparator in ['original','rna_control']:
        delta=scores['protein']-scores[comparator]
        differences['protein_minus_'+comparator]={'mean_mse_difference':float(delta.mean()),'fraction_cells_lower_mse':float((delta<0).mean()),'interpretation':'Negative difference favors protein for this reconstruction diagnostic only'}
    report={'scope':'Paired held-out-from-continued-training RNA reconstruction diagnostic, NOT downstream single-cell performance or full paper reproduction','checkpoint_sha256':hashes,'evaluation_inputs_sha256':sha(OUT/'evaluation_inputs.npz'),'cells':1024,'mask_seed':73021,'comparisons':differences,'limits':'Original pretraining overlap unknown; no donor/study split; one training seed; RNA count ties differ from continuous LFQ. Masked CLS outputs are not standard downstream embeddings.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summaries,indent=2));print(json.dumps(differences,indent=2));print(OUT.resolve())

run()
