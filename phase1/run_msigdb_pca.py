"""Independent MSigDB PCA reconstruction plus the existing MLP evaluation.
Save beside run_msigdb_models.py. Existing files and results are not changed.
Default: build 512-dimensional PCA gene loadings from 50,000 sampled cells,
then run 72 MLP fits. --build-only skips training; --smoke trains three epochs.
This is a documented reconstruction choice, NOT an author-equivalent PCA.
"""
from pathlib import Path
import sys, json, hashlib, gzip, shutil
import numpy as np
import pandas as pd
import h5py
from scipy import sparse, linalg
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score
import run_msigdb_models as existing

OUT = Path(__file__).resolve().parent / 'results/msigdb_pca_reconstruction'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def labels(base):
    with h5py.File(base/'gene_embs/embs_adata.h5ad.gz', 'r') as f:
        genes=existing.frame_index(f['obs']); sigs=existing.frame_index(f['var'])
        Y=existing.matrix(f['X']).astype(np.float32)
    if Y.shape != (len(genes),len(sigs)) or not np.isin(Y,[0,1]).all():
        raise ValueError('Invalid binary label matrix')
    if len(set(genes))!=len(genes) or len(set(sigs))!=len(sigs):
        raise ValueError('Duplicate label identifiers')
    memberships=set()
    with (base/'sigs/genesets.v2023.2.Hs_hallmark.gmt').open(encoding='utf-8-sig') as f:
        for line in f:
            p=line.rstrip('\r\n').split('\t')
            if len(p)<3: raise ValueError('Malformed GMT')
            memberships.update((g,p[0]) for g in p[2:])
    expected=np.array([[(g,s) in memberships for s in sigs] for g in genes],dtype=np.float32)
    if not np.array_equal(Y,expected): raise ValueError('GMT labels disagree')
    return genes,sigs,Y


def resolve_feature_symbols(symbols, ids, genes):
    # Resolve only missing exact symbols; never substitute antisense/prefix matches.
    from urllib.request import Request, urlopen
    from urllib.parse import quote
    resolved = list(symbols)
    stable_ids = [str(v).split('.')[0] for v in ids]
    missing = sorted(set(genes)-set(symbols))
    records = []
    for gene in missing:
        url = 'https://rest.ensembl.org/lookup/symbol/homo_sapiens/' + quote(str(gene), safe='') + '?content-type=application/json'
        request = Request(url, headers={'Accept':'application/json','Content-Type':'application/json'})
        with urlopen(request, timeout=60) as response:
            evidence = json.load(response)
        if evidence.get('display_name') != gene or evidence.get('species') != 'homo_sapiens' or evidence.get('object_type') != 'Gene':
            raise ValueError('Unverified symbol resolution: '+str(gene))
        stable_id = evidence.get('id','')
        hits = [i for i,v in enumerate(stable_ids) if v == stable_id]
        record = dict(gene=str(gene), ensembl_id=stable_id, source_rows=hits,
                      original_names=[str(symbols[i]) for i in hits], url=url, evidence=evidence)
        records.append(record)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'gene_resolution.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        if len(hits) != 1:
            raise ValueError('Exact Ensembl ID for '+str(gene)+' has '+str(len(hits))+' source rows; see gene_resolution.json. No genes were dropped or imputed.')
        i = hits[0]
        if resolved[i] in set(genes) and resolved[i] != gene:
            raise ValueError('Conflicting benchmark symbol for '+stable_id)
        resolved[i] = gene
        print('Resolved:',gene,stable_id,'source name:',symbols[i],flush=True)
    return resolved, records


def expression_rows(path, genes, max_cells=50000):
    # Stream CSR rows, never densify the complete 524k x 60k matrix.
    with h5py.File(path,'r') as f:
        symbols=existing.strings(f['var/feature_name'])
        ids=existing.strings(f['var/feature_id'])
        node=f['X']
        if isinstance(node,h5py.Dataset): shape=node.shape; encoding='dense'
        else:
            encoding=node.attrs.get('encoding-type','')
            if isinstance(encoding,bytes): encoding=encoding.decode()
            shape=tuple(node.attrs['shape'])
            if encoding!='csr_matrix':
                raise ValueError('Source X must be dense or CSR; found '+str(encoding))
        if len(shape)!=2 or shape[1]!=len(symbols): raise ValueError('Expression schema mismatch')
        lookup={g:i for i,g in enumerate(genes)}
        resolved_symbols, resolution = resolve_feature_symbols(symbols, ids, genes)
        cols=np.array([i for i,g in enumerate(resolved_symbols) if g in lookup],dtype=int)
        mapped=np.array([lookup[resolved_symbols[i]] for i in cols],dtype=int)
        mapping=sparse.csr_matrix((np.ones(len(cols)),(cols,mapped)),shape=(shape[1],len(genes)))
        selected=np.sort(np.random.RandomState(42).choice(shape[0],min(max_cells,shape[0]),replace=False))
        blocks=[]; used=[]; dropped=[]
        for start in range(0,shape[0],512):
            take=selected[(selected>=start)&(selected<start+512)]
            if not len(take): continue
            stop=min(start+512,shape[0])
            if encoding=='dense': block=sparse.csr_matrix(node[start:stop])
            else:
                ptr=node['indptr'][start:stop+1]; lo,hi=int(ptr[0]),int(ptr[-1])
                block=sparse.csr_matrix((node['data'][lo:hi],node['indices'][lo:hi],ptr-lo),shape=(stop-start,shape[1]))
            block=block[take-start].astype(np.float64)
            if not np.isfinite(block.data).all() or (block.data<0).any():
                raise ValueError('Source X has negative or nonfinite values')
            if not np.allclose(block.data,np.rint(block.data),rtol=0,atol=1e-5):
                raise ValueError('Source X is not integer-like counts; refusing to normalize potentially transformed data')
            totals=np.asarray(block.sum(axis=1)).ravel()
            keep=totals>0
            dropped.extend(take[~keep].tolist()); used.extend(take[keep].tolist())
            # Library sizes use ALL source features, not only benchmark genes.
            projected=(block[keep]@mapping).tocsr()
            projected=projected.multiply((10000/totals[keep])[:,None]).tocsr()
            projected.data=np.log1p(projected.data)
            blocks.append(projected)
            if len(used)%5000<512: print('Expression cells read:',len(used),flush=True)
        X=sparse.vstack(blocks,format='csr')
        info=dict(source_shape=[int(v) for v in shape],source_encoding=encoding,sampled_cells=len(selected),used_cells=len(used),zero_library_cells=len(dropped),evaluation_genes=len(genes),duplicate_symbol_rows_summed=int(len(cols)-len(genes)),integer_like_sampled_counts=True)
        mapping_table=pd.DataFrame({'feature_id':ids[cols],'feature_name':symbols[cols],'benchmark_gene_index':mapped,'resolved_gene':np.asarray(resolved_symbols)[cols]})
    return X,np.array(used),mapping_table,info


def centered_pca(X,k=512,power_iterations=2):
    # Randomized SVD of centered X; centering is implicit to preserve sparsity.
    n,p=X.shape
    if k>=min(n,p): raise ValueError('Insufficient samples/features for requested PCA dimension')
    mean=np.asarray(X.mean(axis=0)).ravel()
    def dot(B): return np.asarray(X@B)-(mean@B)[None,:]
    def tdot(B): return np.asarray(X.T@B)-mean[:,None]*B.sum(axis=0)[None,:]
    width=min(k+20,min(n,p))
    omega=np.random.RandomState(42).normal(size=(p,width))
    Q=linalg.qr(dot(omega),mode='economic')[0]
    for _ in range(power_iterations):
        Z=linalg.qr(tdot(Q),mode='economic')[0]
        Q=linalg.qr(dot(Z),mode='economic')[0]
    B=tdot(Q).T
    _,singular,Vt=linalg.svd(B,full_matrices=False,check_finite=True)
    variance=singular[:k]**2/(n-1)
    components=Vt[:k]
    # Deterministic component signs for repeatable exported loadings.
    sign=np.sign(components[np.arange(k),np.abs(components).argmax(axis=1)])
    components*=np.where(sign==0,1,sign)[:,None]
    loadings=(components.T*np.sqrt(variance)).astype(np.float32)
    total_variance=float((X.multiply(X).sum()-n*np.dot(mean,mean))/(n-1))
    if not np.isfinite(loadings).all() or total_variance<=0: raise ValueError('Invalid PCA')
    return loadings,variance,float(variance.sum()/total_variance)


def build(base,out,genes,k=512,max_cells=50000):
    source=base/'cellxgene_ds_data/rep_1.h5ad.gz'
    if not source.is_file(): raise FileNotFoundError(source)
    identity=dict(source=str(source.resolve()),source_bytes=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,artifact_sha256=sha256(base/'gene_embs/embs_adata.h5ad.gz'),gmt_sha256=sha256(base/'sigs/genesets.v2023.2.Hs_hallmark.gmt'),components=k,max_cells=max_cells,sampling_seed=42,svd_seed=42,power_iterations=2,preprocessing='Sum duplicate gene-symbol counts; normalize using all-feature library size to 10000; natural log1p; center genes; no variance scaling',pca_feature_universe='Only the deposited benchmark gene universe; includes held-out genes without labels (transductive)',scope='Independent reconstruction; author PCA preprocessing/dimension not established')
    cache=out/'pca_loadings.npz'; manifest=out/'pca_identity.json'
    if cache.exists():
        if not manifest.exists() or json.loads(manifest.read_text())!=identity:
            raise ValueError('Cached PCA belongs to different inputs/settings')
        with np.load(cache,allow_pickle=False) as f:
            if not np.array_equal(f['genes'],genes): raise ValueError('Cached gene order differs')
            X=f['loadings']
        if X.shape!=(len(genes),k) or not np.isfinite(X).all(): raise ValueError('Invalid cached loadings')
        return X
    out.mkdir(parents=True,exist_ok=True)
    actual=source
    with source.open('rb') as f: magic=f.read(2)
    if magic==b'\x1f\x8b':
        actual=out/'expression_uncompressed.h5ad'
        with gzip.open(source,'rb') as src,actual.open('wb') as dest: shutil.copyfileobj(src,dest,8*1024*1024)
    print('Building independent PCA: sampled cells',max_cells,'components',k,flush=True)
    with threadpool_limits(limits=8):
        X,rows,mapping,info=expression_rows(actual,genes,max_cells)
        loadings,variance,ratio=centered_pca(X,k)
    np.savez_compressed(cache,genes=genes.astype(str),loadings=loadings,explained_variance=variance)
    pd.DataFrame({'source_row':rows}).to_csv(out/'pca_cells.csv',index=False)
    mapping.to_csv(out/'pca_gene_mapping.csv',index=False)
    info.update(explained_variance_ratio=ratio,loadings_shape=list(loadings.shape),source_identity_limit='Source size/mtime recorded; source file not fully SHA256 hashed',loadings_sha256=sha256(cache))
    (out/'pca_build.json').write_text(json.dumps(info,indent=2))
    manifest.write_text(json.dumps(identity,indent=2))
    if actual!=source: actual.unlink()
    print('PCA built:',loadings.shape,'explained variance:',ratio,flush=True)
    return loadings


def evaluate(X,genes,sigs,Y,out,smoke=False):
    import torch
    torch.set_num_threads(8)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=out.parent/(out.name+'_smoke') if smoke else out
    out.mkdir(parents=True,exist_ok=True)
    seeds=[0] if smoke else [0,1,2];configs=existing.GRID[:1] if smoke else existing.GRID
    identity=dict(loadings_sha256=sha256(OUT/'pca_loadings.npz'),seeds=seeds,configs=configs,smoke=smoke,torch=torch.__version__,device=str(device))
    manifest=out/'run_identity.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity: raise ValueError('Output identity differs')
    manifest.write_text(json.dumps(identity,indent=2))
    pd.DataFrame(Y,index=pd.Index(genes,name='gene'),columns=sigs).to_csv(out/'labels.csv')
    records=[]
    for seed in seeds:
        parts=existing.partitions(len(genes),seed);sd=out/f'seed_{seed}';sd.mkdir(exist_ok=True)
        partition_labels=np.empty(len(genes),dtype=object)
        for name,ix in parts.items(): partition_labels[ix]=name
        pd.DataFrame({'gene':genes,'partition':partition_labels}).to_csv(sd/'partitions.csv',index=False)
        md=sd/'pca_reconstructed';md.mkdir(exist_ok=True)
        scores=[];best_score=float('inf');selected=None;best_config=None
        for ci,config in enumerate(configs):
            print('PCA seed',seed,'configuration',ci+1,'/',len(configs),config,flush=True)
            model,score,history,best_epoch,state=existing.fit_config(torch,X,Y,parts,config,seed,device,3 if smoke else 10000)
            pd.DataFrame(history).to_csv(md/f'config_{ci:02d}_history.csv',index=False)
            scores.append({**config,'test_log_loss':score,'best_epoch':best_epoch,'epochs':len(history)})
            pd.DataFrame(scores).to_csv(md/'config_scores.csv',index=False)
            if score<best_score:
                best_score=score;best_config=config.copy()
                selected=existing.predict(torch,model,X[parts['benchmark']],device)
                torch.save({'state_dict':state,'config':config,'input_dim':X.shape[1],'sig_names':sigs.tolist(),'seed':seed},md/'best_model.pt')
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
        pd.DataFrame(selected,index=pd.Index(genes[parts['benchmark']],name='gene'),columns=sigs).to_csv(md/'benchmark_predictions.csv')
        ap=float(average_precision_score(Y[parts['benchmark']].ravel(),selected.ravel()))
        records.append(dict(seed=seed,embedding='pca_reconstructed',pooled_auprc=ap,benchmark_genes=len(parts['benchmark']),selected_test_log_loss=best_score,**best_config,scope='smoke_only' if smoke else 'independent_PCA_reconstruction'))
        pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
        print('Completed PCA seed',seed,'pooled AUPRC',ap,flush=True)
    results=pd.DataFrame(records)
    results.groupby('embedding').agg(mean_auprc=('pooled_auprc','mean'),sd_auprc=('pooled_auprc','std'),seeds=('seed','count')).reset_index().to_csv(out/'summary.csv',index=False)
    notes=dict(scope='Independent MSigDB PCA reconstruction; not exact author PCA',preserved='Calls unchanged existing fit_config, predict, partitions and GRID; same labels/order, three seeds, epoch selection, test-loss configuration selection, pooled benchmark AP',departures='50k cells sampled seed42; 512 PCA loadings; benchmark-gene feature universe; library normalization/log1p; transductive unsupervised PCA; all are reconstruction choices, not verified author settings',existing_runner_sha256=sha256(Path(existing.__file__)),smoke=smoke)
    (out/'assumptions.json').write_text(json.dumps(notes,indent=2))
    print(results.to_string(index=False))


def main():
    args=sys.argv[1:]
    if any(a not in ('--smoke','--build-only') for a in args): raise SystemExit('Options: --smoke or --build-only')
    candidates=[Path(r'D:\data\phase1\benchmarks\msigdb'),existing.BASE]
    base=next((p for p in candidates if (p/'cellxgene_ds_data/rep_1.h5ad.gz').is_file() and (p/'gene_embs/embs_adata.h5ad.gz').is_file()),None)
    if base is None: raise FileNotFoundError('MSigDB expression/label inputs not found under D: or existing runner BASE')
    genes,sigs,Y=labels(base)
    if Y.shape!=(3874,50): raise ValueError('Expected verified 3874 x 50 gene universe')
    X=build(base,OUT,genes)
    if '--build-only' not in args: evaluate(X,genes,sigs,Y,OUT,smoke='--smoke' in args)


if __name__=='__main__':
    main()
