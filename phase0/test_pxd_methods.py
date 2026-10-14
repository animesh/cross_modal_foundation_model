"""Real-data method audit for PXD014791. Not an exact reproduction.
Run in phase0: python test_pxd_methods.py
Uses 308 formal samples only. TRS exclusion is provisional, not an identity claim.
All choices are labelled. No best-match tuning or arbitrary pair removal.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
warnings.filterwarnings("ignore","Mean of empty slice")
ROOT=Path(__file__).resolve().parent

def exact_pca(a,q=None):
    center=a.mean(axis=0)
    x=a-center
    eigenvalues,u=np.linalg.eigh(x@x.T)
    order=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[order];u=u[:,order]
    keep=eigenvalues>max(eigenvalues[0]*1e-12,1e-12)
    singular=np.sqrt(eigenvalues[keep]);u=u[:,keep]
    train=u*singular
    query=None if q is None else ((q-center)@x.T@u)/singular
    return train,query

def run(data=None,out=None):
    data=Path(data) if data else Path(r'D:\data\phase0\data\PXD014791')
    out=Path(out) if out else ROOT/'results/method_audit'
    out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_excel(data/'Metadata-Table.xlsx',engine='openpyxl')
    meta['key']=meta['Raw Mass Spectrometry File'].str.replace(r'\.raw$','',regex=True,case=False).str.strip().str.lower()
    assert len(meta)==308 and not meta.key.duplicated().any()
    pg=pd.read_csv(data/'proteinGroups.txt',sep='\t',low_memory=False)
    for col in ['Only identified by site','Reverse','Potential contaminant']:
        if col in pg:pg=pg[pg[col].fillna('').ne('+')]
    pg['gene']=pg['Gene names'].fillna('').str.split(';').str[0].str.strip()
    pg=pg[pg.gene.ne('')].drop_duplicates('gene')
    lookup={col[len('LFQ intensity '):].strip().lower():col for col in pg if col.startswith('LFQ intensity ')}
    assert set(meta.key)<=set(lookup)
    intensity=pg[[lookup[k] for k in meta.key]].to_numpy(float).T
    assert not np.isinf(intensity).any() and not (intensity<0).any()
    intensity=np.where(np.isfinite(intensity),intensity,0)
    ctrl=meta['Drug Treatment'].str.upper().isin(['CTRL','CTROL','CONTROL','DMSO','VEH']).to_numpy()
    assert ctrl.sum()==71
    groups=list(meta[~ctrl].groupby(['Drug Treatment','Cell Line']).groups.items())
    assert len(groups)==60
    ids=[np.array(list(ix)) for _,ix in groups]
    drugs=np.array([key[0] for key,_ in groups]);contexts=np.array([key[1] for key,_ in groups])
    assert (drugs=='TRS').sum()==2 and (drugs=='TRA').sum()==1
    records=[]
    def average(v,idx,method):
        a=v[idx]
        if method=='mean_log_detected':return np.nanmean(np.where(a>0,np.log2(np.maximum(a,1e-300)),np.nan),axis=0)
        if method=='log_mean_detected':
            m=np.nanmean(np.where(a>0,a,np.nan),axis=0)
            return np.log2(m)
        if method=='mean_log1p_zero':return np.log2(a+1).mean(axis=0)
        raise ValueError(method)
    def correlations(pred,truth,mask):
        mask=mask&np.isfinite(pred)&np.isfinite(truth)
        x=pred[mask];y=truth[mask]
        if len(x)<2:return np.nan,np.nan,len(x)
        top=np.argsort(np.abs(y))[-min(100,len(y)):]
        def corr(a,b):return float(pearsonr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan
        return corr(x,y),corr(x[top],y[top]),len(x)
    def weighted(v,w):
        finite=np.isfinite(v);den=(finite*w[:,None]).sum(axis=0)
        return np.divide(np.nansum(v*w[:,None],axis=0),den,out=np.full(v.shape[1],np.nan),where=den>0)
    def predict(q,t,response):
        den=np.linalg.norm(t,axis=1)*np.linalg.norm(q)
        sims=np.divide(t@q,den,out=np.zeros(len(t)),where=den>0)
        ix=np.argsort(sims)[-5:];w=np.maximum(sims[ix],0)
        if not w.sum():w=np.ones(len(ix))
        return weighted(response[ix],w/w.sum())
    def evaluate(pred,r,c,config,i):
        # Two explicit conventions: finite LFC intersection; union with missing log values zero.
        for scoring in ['finite_intersection','detected_union_missing_log_zero']:
            if scoring=='finite_intersection':
                truth=r-c;estimate=pred-c;mask=np.ones(len(r),bool)
            else:
                mask=(r>0)|(pred>0) if config['aggregation']=='mean_log1p_zero' else np.isfinite(r)|np.isfinite(pred)
                truth=np.nan_to_num(r,nan=0)-np.nan_to_num(c,nan=0)
                estimate=np.nan_to_num(pred,nan=0)-np.nan_to_num(c,nan=0)
            full,top,n=correlations(estimate,truth,mask)
            records.append(dict(**config,scoring=scoring,drug=drugs[i],context=contexts[i],full=full,top100=top,n_scored=n))
    # Contract tests for missing-neighbour normalization and Pearson invariance.
    assert np.allclose(weighted(np.array([[10.,np.nan],[20.,30.]]),np.array([.5,.5])),[15,30])
    x=np.arange(5.);y=np.array([0.,2.,1.,4.,3.])
    assert np.isclose(pearsonr(x,y).statistic,pearsonr(x+20,y-10).statistic)
    for aggregation in ['mean_log_detected','log_mean_detected','mean_log1p_zero']:
        r=np.stack([average(intensity,ix,aggregation) for ix in ids])
        c=np.stack([average(intensity,np.flatnonzero(ctrl&meta['Cell Line'].eq(ctx).to_numpy()),aggregation) for ctx in contexts])
        d=r-c
        for i in range(60):
            pool=np.arange(60)!=i
            p=np.nanmean(r[pool],axis=0)
            config=dict(aggregation=aggregation,preprocessing='raw',embedding='none',scaling='none',baseline='perturbation-mean',pca_components=0)
            evaluate(p,r[i],c[i],config,i)
            g=np.nanmean(d[pool],axis=0)
            dm=drugs[pool]==drugs[i];cm=contexts[pool]==contexts[i]
            effect=(np.nanmean(d[pool][dm],axis=0) if dm.any() else np.zeros_like(g))+(np.nanmean(d[pool][cm],axis=0) if cm.any() else np.zeros_like(g))-g
            config['baseline']='additive-linear';evaluate(c[i]+effect,r[i],c[i],config,i)
        print('Completed non-embedding:',aggregation,flush=True)
        # Sample-level representations use the same transformation for all samples.
        sample_log=np.log2(intensity+1) if aggregation=='mean_log1p_zero' else np.where(intensity>0,np.log2(np.maximum(intensity,1e-300)),np.nan)
        for preprocessing in ['raw','sample_median_centered']:
            sl=sample_log.copy()
            if preprocessing=='sample_median_centered':sl=sl-np.nanmedian(sl,axis=1)[:,None]
            # Response targets remain original log-proteomes, not imputed/centred values.
            for embedding in ['sample_global','sample_fold','pair_fold']:
                for scaling in ['none','standardized']:
                    global_cache=None
                    if embedding=='sample_global':
                        fit=sl;mask=np.isfinite(fit).any(axis=0);means=np.nanmean(fit[:,mask],axis=0)
                        a=np.where(np.isfinite(fit[:,mask]),fit[:,mask],means)
                        center=a.mean(axis=0);scale=a.std(axis=0);scale[scale==0]=1
                        if scaling=='standardized':a=(a-center)/scale
                        ncomp=min(512,len(a)-1,a.shape[1])
                        with threadpool_limits(limits=1):
                            pcs,_=exact_pca(a)
                        global_cache=(a,pcs,ncomp)
                    for i in range(60):
                        pool=np.arange(60)!=i
                        if embedding=='sample_global':
                            a,pcs,ncomp=global_cache
                            # Select HVGs by sample variance; normalize individual sample vectors first.
                            hvg=np.argsort(np.var(a,axis=0))[-min(2000,a.shape[1]):]
                            ha=a[:,hvg];norm=np.linalg.norm(ha,axis=1,keepdims=True)
                            ha=np.divide(ha,norm,out=np.zeros_like(ha),where=norm>0)
                            he=np.stack([ha[ix].mean(axis=0) for ix in ids])
                            pe=np.stack([pcs[ix].mean(axis=0) for ix in ids])
                            hq,ht=he[i],he[pool];pq,pt=pe[i],pe[pool]
                        else:
                            if embedding=='sample_fold':
                                held=np.zeros(len(meta),bool);held[ids[i]]=True
                                fit=sl[~held];query=sl[held]
                            else:
                                pair_sl=np.stack([np.nanmean(sl[ix],axis=0) for ix in ids])
                                fit=pair_sl[pool];query=pair_sl[i:i+1]
                            mask=np.isfinite(fit).any(axis=0);means=np.nanmean(fit[:,mask],axis=0)
                            a=np.where(np.isfinite(fit[:,mask]),fit[:,mask],means)
                            q=np.where(np.isfinite(query[:,mask]),query[:,mask],means)
                            center=a.mean(axis=0);scale=a.std(axis=0);scale[scale==0]=1
                            if scaling=='standardized':a=(a-center)/scale;q=(q-center)/scale
                            hvg=np.argsort(np.var(a,axis=0))[-min(2000,a.shape[1]):]
                            ha=a[:,hvg];hq_samples=q[:,hvg]
                            for v in [ha,hq_samples]:
                                norm=np.linalg.norm(v,axis=1,keepdims=True);np.divide(v,norm,out=v,where=norm>0)
                            ncomp=min(512,len(a)-1,a.shape[1])
                            with threadpool_limits(limits=1):
                                pcs,qpcs=exact_pca(a,q)
                            if embedding=='sample_fold':
                                sample_indices=np.flatnonzero(~held);index_map={original:new for new,original in enumerate(sample_indices)}
                                pool_ids=[np.array([index_map[j] for j in ids[k]]) for k in np.flatnonzero(pool)]
                                ht=np.stack([ha[ix].mean(axis=0) for ix in pool_ids]);pt=np.stack([pcs[ix].mean(axis=0) for ix in pool_ids])
                                hq=hq_samples.mean(axis=0);pq=qpcs.mean(axis=0)
                            else:ht=ha;pt=pcs;hq=hq_samples[0];pq=qpcs[0]
                        for baseline,q,t in [('HVG',hq,ht),('PCA',pq,pt)]:
                            pred=predict(q,t,r[pool])
                            config=dict(aggregation=aggregation,preprocessing=preprocessing,embedding=embedding,scaling=scaling,baseline=baseline,pca_components=ncomp if baseline=='PCA' else 0)
                            evaluate(pred,r[i],c[i],config,i)
                    print('Completed',aggregation,preprocessing,embedding,scaling,flush=True)
    result=pd.DataFrame(records)
    result.to_csv(out/'per_pair.csv',index=False)
    keys=['aggregation','preprocessing','embedding','scaling','baseline','scoring']
    summaries=[]
    for scenario,sub in [('all_60',result),('provisional_exclude_TRS_58',result[result.drug!='TRS'])]:
        summary=sub.groupby(keys).agg(full_mean=('full','mean'),top100_mean=('top100','mean'),n_pairs=('drug','size'),pca_min=('pca_components','min'),pca_max=('pca_components','max')).reset_index()
        summary.insert(0,'scenario',scenario);summaries.append(summary)
    summary=pd.concat(summaries,ignore_index=True)
    summary.to_csv(out/'summary.csv',index=False)
    assert not result.duplicated(keys+['drug','context']).any()
    assert result.n_scored.min()>0
    assert set(summary.n_pairs)=={58,60}
    # Independent reaggregation of all means.
    for _,row in summary.iterrows():
        sub=result
        for key in keys:sub=sub[sub[key]==row[key]]
        if row.scenario!='all_60':sub=sub[sub.drug!='TRS']
        assert np.isclose(sub.full.mean(),row.full_mean,equal_nan=True)
        assert np.isclose(sub.top100.mean(),row.top100_mean,equal_nan=True)
    notes={'status':'method sensitivity, not exact reproduction','data':'308 formal samples; 60 pairs; original first-gene deduplication preserved','experiments':'3 aggregation conventions x 2 scoring conventions; raw/sample median; sample global/sample fold/pair fold; scaled/unscaled; HVG/PCA; mean/additive baselines','sample_global':'Transductive fit on all 308 samples including query; explicitly labelled, not leakage-free','sample_fold':'Fit excludes held-out treated samples; controls and other treated samples included','pair_fold':'Fit excludes held-out pair mean','PCA':'512 requested, capped to available centred sample rank; sample global 307; fold variable; pair fold 58','missing_union':'Missing log abundance explicitly zero-filled; a sensitivity assumption, not a verified author convention','log_mean_detected_embeddings':'Embeddings still constructed from individual log samples; response aggregation is log(mean intensity)','TRS':'No identity assigned; exclusion provisional','targets':'Do not optimize against unverified approximate values in previous code','checked':'Unique records, 58/60 counts, all means independently reaggregated, missing-neighbour normalization, correlation offset invariance','runtime':'Executed in the environment running this script; no author-container validation claimed'}
    (out/'assumptions.json').write_text(json.dumps(notes,indent=2))
    print('Records',len(result),'summaries',len(summary),flush=True)
    return summary

if __name__=='__main__':run()
