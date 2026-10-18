"""Sequential MSigDB multilabel MLP on deposited embeddings; PCA excluded.
Save in phase1. First: python run_msigdb_models.py --smoke
Full run: python run_msigdb_models.py
Only smoke is a diagnostic, never a full benchmark result.
"""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.metrics import average_precision_score,log_loss
BASE=Path(r'L:\data\phase1\benchmarks\msigdb')
OUT=Path(__file__).resolve().parent/'results/msigdb_models'
GRID=[dict(hidden_size=h,n_hidden_layers=n,dropout=d) for h,n,d in itertools.product([64,512,1024,2048],[1,2],[0.0,0.1,0.5])]

def strings(node):
    if isinstance(node,h5py.Group):
        codes=node['codes'][()]
        if (codes<0).any():raise ValueError('Missing categorical values')
        return strings(node['categories'])[codes]
    a=node[()]
    return np.array([v.decode() if isinstance(v,bytes) else str(v) for v in a])

def frame_index(group):
    key=group.attrs.get('_index','_index')
    if isinstance(key,bytes):key=key.decode()
    if key not in group:raise ValueError('Missing H5AD dataframe index')
    return strings(group[key])

def matrix(node):
    if isinstance(node,h5py.Dataset):return node[()]
    encoding=node.attrs.get('encoding-type','')
    if isinstance(encoding,bytes):encoding=encoding.decode()
    cls={'csr_matrix':sparse.csr_matrix,'csc_matrix':sparse.csc_matrix}.get(encoding)
    if cls is None:raise ValueError('Unsupported matrix encoding '+str(encoding))
    return cls((node['data'][()],node['indices'][()],node['indptr'][()]),shape=tuple(node.attrs['shape'])).toarray()

def read_inputs(base):
    path=base/'gene_embs/embs_adata.h5ad.gz'
    with h5py.File(path,'r') as f:
        genes=frame_index(f['obs']);sigs=frame_index(f['var']);Y=matrix(f['X']).astype(np.float32)
        if Y.shape!=(len(genes),len(sigs)) or not np.isin(Y,[0,1]).all():raise ValueError('Invalid binary hit matrix')
        if len(set(genes))!=len(genes) or len(set(sigs))!=len(sigs):raise ValueError('Duplicate gene/signature IDs')
        embeddings={}
        for name,node in f['obsm'].items():
            if isinstance(node,h5py.Group) and node.attrs.get('encoding-type') in ['dataframe',b'dataframe']:
                if not np.array_equal(frame_index(node),genes):raise ValueError('Embedding gene order differs: '+name)
                columns=node.attrs['column-order']
                X=np.column_stack([node[c.decode() if isinstance(c,bytes) else c][()] for c in columns])
            else:X=matrix(node)
            X=np.asarray(X,dtype=np.float32)
            if X.shape[0]!=len(genes) or not np.isfinite(X).all():raise ValueError('Invalid embedding '+name)
            embeddings[name]=X
    # Verify deposited binary memberships against the GMT, not only its shape.
    members=set()
    with (base/'sigs/genesets.v2023.2.Hs_hallmark.gmt').open(encoding='utf-8-sig') as f:
        for line in f:
            parts=line.rstrip('\r\n').split('\t')
            if len(parts)<3:raise ValueError('Malformed GMT')
            members.update((g,parts[0]) for g in parts[2:])
    expected=np.array([[(g,s) in members for s in sigs] for g in genes],dtype=np.float32)
    if not np.array_equal(expected,Y):raise ValueError('Deposited labels disagree with GMT')
    return genes,sigs,Y,embeddings

def partitions(n,seed):
    cuts=(np.cumsum([.7,.05,.05])*n).astype(int)
    idx=np.random.RandomState(seed).permutation(n)
    parts=dict(zip(['train','val','test','benchmark'],np.split(idx,cuts)))
    if any(len(v)==0 for v in parts.values()):raise ValueError('Empty partition')
    if len(set(np.concatenate(list(parts.values()))))!=n:raise ValueError('Overlapping partitions')
    return parts

def make_model(torch,dim,outputs,config):
    nn=torch.nn;h=config['hidden_size'];d=config['dropout']
    layers=[nn.Linear(dim,h),nn.LeakyReLU()]
    for _ in range(config['n_hidden_layers']-1):layers.extend([nn.Dropout(d),nn.Linear(h,h),nn.LeakyReLU()])
    layers.extend([nn.Dropout(d),nn.Linear(h,outputs),nn.Sigmoid()])
    return nn.Sequential(*layers)

def predict(torch,model,X,device):
    model.eval();chunks=[]
    with torch.no_grad():
        for start in range(0,len(X),256):chunks.append(model(torch.from_numpy(X[start:start+256]).to(device)).cpu().numpy())
    result=np.concatenate(chunks)
    if not np.isfinite(result).all() or ((result<0)|(result>1)).any():raise ValueError('Invalid probabilities')
    return result

def fit_config(torch,X,Y,parts,config,seed,device,max_epochs):
    torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=make_model(torch,X.shape[1],Y.shape[1],config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=7)
    ds=torch.utils.data.TensorDataset(torch.from_numpy(X[parts['train']]),torch.from_numpy(Y[parts['train']]))
    loader=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True,num_workers=0)
    best_loss=float('inf');best_state=None;best_epoch=0;early_best=float('inf');wait=0;history=[]
    for epoch in range(1,max_epochs+1):
        model.train();total=0.0
        for xb,yb in loader:
            xb=xb.to(device);yb=yb.to(device);optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy(model(xb),yb)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=loss.item()*len(xb)
        # Weighted BCE over validation batches, as the upstream epoch metric.
        model.eval();val_sum=0.0
        with torch.no_grad():
            for start in range(0,len(parts['val']),256):
                ix=parts['val'][start:start+256]
                xb=torch.from_numpy(X[ix]).to(device);yb=torch.from_numpy(Y[ix]).to(device)
                val_sum+=torch.nn.functional.binary_cross_entropy(model(xb),yb).item()*len(ix)
        val_loss=val_sum/len(parts['val'])
        if not np.isfinite(val_loss):raise ValueError('Nonfinite validation loss')
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        # Lightning EarlyStopping defaults: min_delta=0.0, patience=15.
        if val_loss<early_best:early_best=val_loss;wait=0
        else:wait+=1
        scheduler.step(val_loss)
        history.append({'epoch':epoch,'train_loss':total/len(ds),'val_loss':val_loss,'lr':optimizer.param_groups[0]['lr']})
        if epoch%50==0:print(f'  epoch {epoch}, val BCE {val_loss:.6f}',flush=True)
        if wait>=15:break
    model.load_state_dict(best_state)
    test_prob=predict(torch,model,X[parts['test']],device)
    score=float(log_loss(Y[parts['test']].ravel(),test_prob.ravel(),labels=[0,1]))
    return model,score,history,best_epoch,best_state

def run(base=BASE,out=OUT,smoke=False):
    import torch
    torch.set_num_threads(8)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        print('GPU:',torch.cuda.get_device_name(0),'free bytes:',torch.cuda.mem_get_info()[0],flush=True)
    base=Path(base);out=Path(out)
    if smoke:out=out.parent/(out.name+'_smoke')
    out.mkdir(parents=True,exist_ok=True)
    genes,sigs,Y,embeddings=read_inputs(base)
    if not smoke and (Y.shape!=(3874,50) or len(embeddings)!=7):raise ValueError('Deposited input differs from verified 3874x50 / seven-embedding inventory')
    print('Device:',device,'labels:',Y.shape,'models:',list(embeddings),flush=True)
    seeds=[0] if smoke else [0,1,2];configs=GRID[:1] if smoke else GRID
    names=[min(embeddings,key=lambda k:embeddings[k].shape[1])] if smoke else list(embeddings)
    # Immutable run identity prevents silently mixing inputs/settings on reruns.
    identity={'artifact_sha256':hashlib.sha256((base/'gene_embs/embs_adata.h5ad.gz').read_bytes()).hexdigest(),'seeds':seeds,'configs':configs,'smoke':smoke,'torch':torch.__version__,'device':str(device)}
    manifest=out/'run_identity.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Output directory belongs to a different run')
    manifest.write_text(json.dumps(identity,indent=2))
    pd.DataFrame(Y,index=pd.Index(genes,name='gene'),columns=sigs).to_csv(out/'labels.csv')
    records=[]
    for seed in seeds:
        parts=partitions(len(genes),seed);sd=out/f'seed_{seed}';sd.mkdir(exist_ok=True)
        partition_labels=np.empty(len(genes),dtype=object)
        for name,ix in parts.items():partition_labels[ix]=name
        pd.DataFrame({'gene':genes,'partition':partition_labels}).to_csv(sd/'partitions.csv',index=False)
        for name in names:
            md=sd/name;md.mkdir(exist_ok=True);X=embeddings[name];scores=[];best_score=float('inf');selected=None;best_config=None
            print(f'Seed {seed}, {name}: {len(configs)} sequential configurations',flush=True)
            for ci,config in enumerate(configs):
                print(f' config {ci+1}/{len(configs)} {config}',flush=True)
                model,score,history,best_epoch,state=fit_config(torch,X,Y,parts,config,seed,device,3 if smoke else 10000)
                pd.DataFrame(history).to_csv(md/f'config_{ci:02d}_history.csv',index=False)
                scores.append({**config,'test_log_loss':score,'best_epoch':best_epoch,'epochs':len(history)})
                pd.DataFrame(scores).to_csv(md/'config_scores.csv',index=False)
                if score<best_score:
                    best_score=score;best_config=config.copy()
                    selected=predict(torch,model,X[parts['benchmark']],device)
                    torch.save({'state_dict':state,'config':config,'input_dim':X.shape[1],'sig_names':sigs.tolist(),'seed':seed},md/'best_model.pt')
                del model
                if device.type=='cuda':torch.cuda.empty_cache()
            pd.DataFrame(selected,index=pd.Index(genes[parts['benchmark']],name='gene'),columns=sigs).to_csv(md/'benchmark_predictions.csv')
            ap=float(average_precision_score(Y[parts['benchmark']].ravel(),selected.ravel()))
            record={'seed':seed,'embedding':name,'pooled_auprc':ap,'benchmark_genes':len(parts['benchmark']),'selected_test_log_loss':best_score,**best_config,'scope':'smoke_only' if smoke else 'available_embeddings_no_PCA'}
            records.append(record);pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
            print(f'Completed {name}, seed {seed}: pooled AUPRC {ap:.6f}',flush=True)
    results=pd.DataFrame(records)
    results.groupby('embedding').agg(mean_auprc=('pooled_auprc','mean'),sd_auprc=('pooled_auprc','std'),seeds=('seed','count')).reset_index().to_csv(out/'summary.csv',index=False)
    notes={'scope':'Available deposited embeddings only; no PCA, no checkpoint inference, no exact Figure 2 claim','preserved':'H5AD gene order and common gene universe; GMT labels checked; seeds 0/1/2; 70/5/5/20 partitions; 24 upstream configs; batch256; Adam lr.001; sigmoid BCE; LR plateau patience7; early stop patience15; max10000; config selection by test log loss; pooled benchmark AUPRC','departures':'Plain PyTorch sequential loop replaces Lightning/multiprocessing; best weights cloned instead of mutable state_dict reference; explicit CPU/GPU handling; no Lightning sanity-validation pass; no numerical identity claimed','smoke':smoke,'training_validation':'Must be established by local smoke/full execution; host training was not executed during delivery'}
    (out/'assumptions.json').write_text(json.dumps(notes,indent=2))
    return results

if __name__=='__main__':
    if any(a!='--smoke' for a in sys.argv[1:]):raise SystemExit('Only supported option: --smoke')
    run(smoke='--smoke' in sys.argv)
