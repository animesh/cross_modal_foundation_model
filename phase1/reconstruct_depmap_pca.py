"""Reconstruct upstream marginal PCA from raw CCLE counts, then compare/retrain RF.
Run from phase1: python reconstruct_depmap_pca.py
Requires the existing run_depmap_upstream.py in the same directory.
Inputs are read-only; outputs are separate from both validated evaluators.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse
from run_depmap_upstream import read_artifact, metrics
BASE=Path(r'D:\data\phase1\benchmarks\depmap')
OUT=Path(__file__).resolve().parent/'results/depmap_raw_pca'

def prepare_counts(base):
    raw=base/'raw'
    counts=pd.read_csv(raw/'ccle-counts.gct',sep='\t')
    if list(counts.columns[:2])!=['Name','Description']:
        raise ValueError('Expected the verified plain TSV Name/Description header')
    gene_ids=counts.Name.astype(str).str.split('.').str[0]
    if gene_ids.duplicated().any():
        raise ValueError('Duplicate version-stripped Ensembl IDs: upstream mapping requires inspection')
    symbols=counts.Description.to_numpy()
    # Gene-effect data determine the usable samples and gene universe upstream.
    effects=pd.read_csv(raw/'depmap-gene-effects.csv',usecols=[0])
    effect_header=pd.read_csv(raw/'depmap-gene-effects.csv',nrows=0).columns[1:]
    metadata=pd.read_csv(raw/'depmap-metadata.csv')
    metadata=metadata[metadata.ModelID.isin(effects.iloc[:,0])].reset_index(drop=True)
    vocab=pd.read_csv(raw/'gene-mapping.csv').feature_name
    common=set(vocab)&{c.split(' ')[0] for c in effect_header}&set(symbols)
    keep=np.flatnonzero(pd.Series(symbols).isin(common).to_numpy())
    keep=keep[np.argsort(gene_ids.iloc[keep].to_numpy())]
    samples=[c for c in counts.columns[2:] if c in set(metadata.CCLEName)]
    if not samples or not len(keep):raise ValueError('Empty upstream sample/gene intersection')
    # Preserve separate Ensembl rows, sorted IDs, original sample order, float32.
    # No gene-symbol summation, log transform, normalization, or standardization.
    X=counts.iloc[keep][samples].to_numpy(dtype=np.float32).T
    if not np.isfinite(X).all() or (X<0).any():raise ValueError('Invalid raw counts')
    obs=pd.DataFrame({'CCLEName':samples}).merge(metadata,on='CCLEName')
    if len(obs)!=len(samples) or obs.CCLEName.tolist()!=samples:
        raise ValueError('Metadata merge changes sample count/order; inspect upstream contract')
    return X,gene_ids.iloc[keep].to_numpy(),symbols[keep],obs

def reconstruct(X,symbols,genes):
    if min(X.shape)<=15:raise ValueError('Insufficient dimensions for 15-component PCA')
    # Explicit full SVD is deterministic. Upstream uses PCA(n_components=15)
    # with auto solver and no seed, so bitwise equivalence is not promised.
    pca=PCA(n_components=15,svd_solver='full')
    pca.fit(X)
    loadings=pca.components_.T*np.sqrt(pca.explained_variance_)
    first={}
    for i,symbol in enumerate(symbols):first.setdefault(symbol,i)
    missing=set(genes)-set(first)
    if missing:raise ValueError(f'{len(missing)} marginal genes lack raw-count loadings')
    selected=loadings[[first[g] for g in genes]]
    return selected,pca

def compare_features(rebuilt,deposited):
    correlations=[];signs=[]
    for i in range(15):
        corr=float(np.corrcoef(rebuilt[:,i],deposited[:,i])[0,1])
        if not np.isfinite(corr):raise ValueError('Undefined feature correlation')
        correlations.append(corr);signs.append(1 if corr>=0 else -1)
    aligned=rebuilt*np.array(signs)
    denominator=np.linalg.norm(deposited)
    if denominator==0:raise ValueError('Zero deposited feature norm')
    return pd.DataFrame({'component':np.arange(1,16),'correlation':correlations,'sign':signs}),float(np.linalg.norm(aligned-deposited)/denominator)

def run(base=BASE,out=OUT):
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    required=['raw/ccle-counts.gct','raw/depmap-gene-effects.csv','raw/depmap-metadata.csv','raw/gene-mapping.csv','raw/depmap-gene-dependencies.csv','misc/split-genes-lt5gt70.csv','gene-embs/pcloading-lt5gt70-bin.h5ad']
    missing=[r for r in required if not (base/r).is_file()]
    if missing:raise FileNotFoundError('Missing inputs: '+', '.join(missing))
    deposited,genes,y=read_artifact(base/required[-1])
    if sparse.issparse(deposited):deposited=deposited.toarray()
    split=pd.read_csv(base/'misc/split-genes-lt5gt70.csv',dtype=str)
    if not split.gene.is_unique or set(split.gene)!=set(genes):raise ValueError('Artifact/split mismatch')
    split=split.set_index('gene').loc[genes]
    dep=pd.read_csv(base/'raw/depmap-gene-dependencies.csv').iloc[:,1:]
    dep.columns=[c.split(' ')[0] for c in dep.columns]
    if not dep.columns.is_unique:raise ValueError('Duplicate dependency gene symbols')
    vals=dep.loc[:,genes].to_numpy(float)
    if np.isinf(vals).any():raise ValueError('Infinite dependency labels')
    finite=vals[np.isfinite(vals)]
    if finite.min()<0 or finite.max()>1:raise ValueError('Expected dependency probabilities')
    fraction=(np.nan_to_num(vals,nan=0)>0.5).mean(0)
    if not np.array_equal((fraction>=0.5).astype(int),y):raise ValueError('Reconstructed labels differ from artifact')
    print('Reconstructing upstream count matrix...',flush=True)
    X,ids,symbols,obs=prepare_counts(base)
    pd.DataFrame({'gene_id':ids,'gene':symbols}).to_csv(out/'count_genes.csv',index=False)
    obs[['CCLEName','ModelID']].to_csv(out/'count_samples.csv',index=False)
    print(f'Raw-count PCA: {X.shape[0]} samples x {X.shape[1]} Ensembl features; 15 components',flush=True)
    rebuilt,pca=reconstruct(X,symbols,genes)
    feature_report,relative_error=compare_features(rebuilt,deposited)
    feature_report.to_csv(out/'feature_comparison.csv',index=False)
    pd.DataFrame(rebuilt,index=pd.Index(genes,name='gene'),columns=[f'PC{i+1}' for i in range(15)]).to_csv(out/'reconstructed_loadings.csv')
    print(f'Sign-aligned relative feature error: {relative_error:.8g}',flush=True)
    rows=[];predictions=[];seen=[]
    for fold in range(5):
        col=f'fold-{fold}'
        if set(split[col])!={'train','val'}:raise ValueError('Invalid fold labels')
        train=(split[col]=='train').to_numpy();val=(split[col]=='val').to_numpy()
        seen.extend(genes[val]);prefix=f'rfc-pcloading-lt5gt70-bin-fold-{fold}-val'
        saved_y=np.load(base/'gene-embs/results'/f'{prefix}-true.npy',allow_pickle=False)
        saved_p=np.load(base/'gene-embs/results'/f'{prefix}-proba.npy',allow_pickle=False)
        if not np.array_equal(saved_y,y[val]) or saved_p.shape!=(val.sum(),2):raise ValueError('Saved RF label/order mismatch')
        rows.append({'fold':fold,'source':'deposited_RF','n_val':int(val.sum()),**metrics(y[val],saved_p[:,1])})
        record=pd.DataFrame({'fold':fold,'gene':genes[val],'score':y[val],'deposited_probability':saved_p[:,1]})
        for source,features in [('deposited_features_seed42',deposited),('raw_reconstructed_seed42',rebuilt)]:
            rf=RandomForestClassifier(n_jobs=8,random_state=42)
            rf.fit(features[train],y[train]);prob=rf.predict_proba(features[val])[:,1]
            rows.append({'fold':fold,'source':source,'n_val':int(val.sum()),**metrics(y[val],prob)})
            record[source]=prob
        predictions.append(record)
        print(f'fold-{fold}: saved {rows[-3]["auprc"]:.6f}; deposited features {rows[-2]["auprc"]:.6f}; reconstructed {rows[-1]["auprc"]:.6f}',flush=True)
    if len(seen)!=len(genes) or len(set(seen))!=len(genes):raise ValueError('Incomplete/disjoint validation coverage')
    result=pd.DataFrame(rows);result.to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    summary=result.groupby('source').agg(mean_auprc=('auprc','mean'),sd_auprc=('auprc','std'),mean_auroc=('auroc','mean'),folds=('fold','count')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    notes={'scope':'Raw-count reconstruction of upstream marginal PCA; not complete target-paper reproduction','counts_shape':list(X.shape),'duplicate_symbol_rows_preserved':int(pd.Series(symbols).duplicated().sum()),'PCA':'Raw float32 counts; sorted version-stripped Ensembl IDs; no scaling/logging; full SVD 15 components; components.T * sqrt(explained_variance)','upstream_difference':'Explicit full solver; upstream auto solver unseeded. Existing gene folds retained, not regenerated. Ambiguous metadata/ID contracts cause a stop.','marginal_gene_selection':'First matching Ensembl row per gene symbol, as upstream','sign_aligned_relative_error':relative_error,'minimum_absolute_component_correlation':float(feature_report.correlation.abs().min()),'explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'labels':'Upstream convention: missing probability -> zero; per-cell >0.5; fraction >=0.5 positive','RF':'Defaults with n_jobs=8 and seed42; no feature rotations or sign alignment used for RF','sklearn_version':sklearn.__version__,'numpy_version':np.__version__,'validation':'Raw data reconstruction and deposited artifact comparison executed by this local run; no author-container equivalence claimed'}
    (out/'assumptions.json').write_text(json.dumps(notes,indent=2))
    print(summary.to_string(index=False));return summary

if __name__=='__main__':run()
