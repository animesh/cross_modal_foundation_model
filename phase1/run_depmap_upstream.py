"""Audit/retrain upstream marginal-essentiality PCA/RF using deposited artifacts.
Save in phase1. Run: python run_depmap_upstream.py
Requires numpy, pandas, scipy, scikit-learn, h5py. Does not modify inputs.
Not raw-data PCA reconstruction or confirmed Figure 2 author configuration.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
import sklearn
BASE=Path(r'D:\data\phase1\benchmarks\depmap')
OUT=Path(__file__).resolve().parent/'results/depmap_upstream'

def vector(node):
    if isinstance(node,h5py.Group):
        if not {'codes','categories'}.issubset(node):raise ValueError('Unsupported H5AD observation encoding')
        codes=node['codes'][()];categories=vector(node['categories'])
        if (codes<0).any():raise ValueError('Missing categorical observations')
        return categories[codes]
    a=node[()]
    if a.dtype.kind in 'SO':return np.array([v.decode() if isinstance(v,bytes) else str(v) for v in a])
    return a

def read_artifact(path):
    with h5py.File(path,'r') as f:
        genes=vector(f['obs/gene']);labels=vector(f['obs/score'])
        node=f['X']
        if isinstance(node,h5py.Dataset):X=node[()]
        else:
            encoding=node.attrs.get('encoding-type','')
            if isinstance(encoding,bytes):encoding=encoding.decode()
            shape=tuple(node.attrs['shape'])
            cls={'csr_matrix':sparse.csr_matrix,'csc_matrix':sparse.csc_matrix}.get(encoding)
            if cls is None:raise ValueError('Unsupported X encoding '+str(encoding))
            X=cls((node['data'][()],node['indices'][()],node['indptr'][()]),shape=shape)
    assert X.shape==(len(genes),15),'Expected upstream 15-component marginal PCA artifact'
    assert len(set(genes))==len(genes) and set(labels)=={0,1}
    assert np.isfinite(X.data if sparse.issparse(X) else X).all()
    return X,genes,labels.astype(int)

def metrics(y,p):
    assert len(y)==len(p) and set(y)=={0,1}
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return {'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p))}

def run(base=BASE,out=OUT):
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    artifact=base/'gene-embs/pcloading-lt5gt70-bin.h5ad'
    splitpath=base/'misc/split-genes-lt5gt70.csv'
    depfile=base/'raw/depmap-gene-dependencies.csv'
    X,genes,y=read_artifact(artifact)
    split=pd.read_csv(splitpath,dtype=str)
    assert split.gene.is_unique and set(split.gene)==set(genes),'Artifact/split gene sets differ'
    split=split.set_index('gene').loc[genes]
    # Match upstream utils.get_marginal_genes_scores exactly: NaN -> 0;
    # per-cell probability > .5; fraction >= .5 -> broadly essential.
    dep=pd.read_csv(depfile).iloc[:,1:]
    dep.columns=[c.split(' ')[0] for c in dep.columns]
    assert dep.columns.is_unique
    assert set(genes).issubset(dep.columns)
    vals=dep.loc[:,genes].to_numpy(float)
    finite=vals[np.isfinite(vals)]
    assert len(finite) and finite.min()>=0 and finite.max()<=1
    fraction=(np.nan_to_num(vals,nan=0)>0.5).mean(axis=0)
    reconstructed=(fraction>=0.5).astype(int)
    assert np.array_equal(y,reconstructed),'Deposited labels disagree with upstream label reconstruction'
    pd.DataFrame({'gene':genes,'score':y,'fraction_dependent':fraction}).to_csv(out/'labels.csv',index=False)
    rows=[];all_predictions=[];seen=[]
    for fold in range(5):
        column=f'fold-{fold}'
        assert set(split[column])=={'train','val'}
        train=(split[column]=='train').to_numpy();val=(split[column]=='val').to_numpy()
        seen.extend(genes[val]);assert not set(genes[train])&set(genes[val])
        prefix=f'rfc-pcloading-lt5gt70-bin-fold-{fold}-val'
        saved_y=np.load(base/'gene-embs/results'/f'{prefix}-true.npy',allow_pickle=False)
        saved_p=np.load(base/'gene-embs/results'/f'{prefix}-proba.npy',allow_pickle=False)
        assert np.array_equal(saved_y,y[val]),'Saved RF labels/order disagree with artifact/split'
        assert saved_p.shape==(val.sum(),2) and np.allclose(saved_p.sum(axis=1),1)
        rows.append({'fold':fold,'source':'deposited_RF','n_val':int(val.sum()),**metrics(saved_y,saved_p[:,1])})
        # Only departure from upstream RF constructor: explicit seed for rerun reproducibility.
        rf=RandomForestClassifier(n_jobs=8,random_state=42)
        rf.fit(X[train],y[train]);assert np.array_equal(rf.classes_,[0,1])
        p=rf.predict_proba(X[val])[:,1]
        rows.append({'fold':fold,'source':'local_RF_seed42','n_val':int(val.sum()),**metrics(y[val],p)})
        all_predictions.append(pd.DataFrame({'fold':fold,'gene':genes[val],'score':y[val],'deposited_probability':saved_p[:,1],'local_probability':p}))
        print(f'fold-{fold}: deposited AUPRC {rows[-2]["auprc"]:.6f}; local AUPRC {rows[-1]["auprc"]:.6f}',flush=True)
    assert len(seen)==len(genes) and len(set(seen))==len(genes)
    result=pd.DataFrame(rows);result.to_csv(out/'metrics.csv',index=False)
    summary=result.groupby('source').agg(mean_auprc=('auprc','mean'),sd_auprc=('auprc','std'),mean_auroc=('auroc','mean'),folds=('fold','count')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    pd.concat(all_predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    notes={'scope':'Upstream marginal-essentiality deposited PCA/RF artifact evaluation and seeded RF retraining; not raw-count PCA reconstruction or exact target-paper configuration',
           'PCA':'15 components; upstream raw-count PCA loadings multiplied by sqrt(explained variance); deposited artifact used without modification',
           'labels':'All dependency rows; NaN filled zero; probability >0.5 discretized; fraction >=0.5 positive; artifact labels checked exactly',
           'split':'Existing five gene folds; each gene validated exactly once; no cell-line fold averaging',
           'RF':'sklearn defaults, n_jobs=8; local random_state=42; upstream random_state unspecified',
           'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'h5py':h5py.__version__},
           'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
           'checks':'Artifact dimensions/labels, exact reconstructed labels, saved RF label order, probability bounds, disjoint folds, complete validation coverage',
           'upstream_git_commit':'Not provided; uploaded source archive inspected, commit provenance unresolved'}
    (out/'assumptions.json').write_text(json.dumps(notes,indent=2))
    print(summary.to_string(index=False));return summary

if __name__=='__main__':run()
