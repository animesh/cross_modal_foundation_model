"""Phase 1 local input preflight. Does not train or claim Figure 2 reproduction.
Run: python check_phase1_data.py
Only pandas/numpy required. Reads expression in chunks, not all into RAM.
Writes a small report and identifier mapping; never modifies benchmark inputs.
"""
from pathlib import Path
import json,sys,re
import numpy as np
import pandas as pd
BENCH=Path(r'D:\data\phase1\benchmarks')
OUT=Path('results/phase1_preflight')

def check(bench=BENCH,out=OUT):
    bench=Path(bench);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'benchmark_root':str(bench),'errors':[],'warnings':[],'files':{},'schemas':{},'scope':'Input preflight only; not evaluation-protocol validation'}
    required=['depmap/raw/ccle-counts.gct','depmap/raw/depmap-gene-dependencies.csv','depmap/misc/split-cls.csv','depmap/misc/split-genes-lt5gt70.csv','msigdb/sigs/genesets.v2023.2.Hs_hallmark.gmt']
    for rel in required:
        p=bench/rel;report['files'][rel]={'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else 0}
        if not p.is_file() or p.stat().st_size==0:report['errors'].append('Missing/empty: '+rel)
    def finish():
        report['input_checks_passed']=not report['errors']
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2));print('Report:',(out/'report.json').resolve())
        return report
    if report['errors']:return finish()
    def split_file(rel,id_col):
        d=pd.read_csv(bench/rel,dtype=str).fillna('');d.columns=d.columns.str.strip()
        report['schemas'][rel]={'rows':len(d),'columns':d.columns.tolist()}
        if id_col not in d:raise ValueError(rel+': missing '+id_col)
        if d[id_col].eq('').any() or d[id_col].duplicated().any():raise ValueError(rel+': blank/duplicate IDs')
        for fold in ['fold-'+str(i) for i in range(5)]:
            if fold not in d:raise ValueError(rel+': missing '+fold)
            labels=d[fold].str.strip().str.lower()
            if not set(labels)<= {'train','val'}:raise ValueError(rel+': unexpected labels '+str(sorted(set(labels))))
            if not {'train','val'}<=set(labels):raise ValueError(rel+': empty train/val partition')
            d[fold]=labels
        return d
    try:
        cls=split_file(required[2],'cell-line');genes=split_file(required[3],'gene')
        gct=bench/required[0]
        with gct.open(encoding='utf-8-sig') as f:
            first=f.readline().rstrip('\r\n')
            if first=='#1.2':
                dims=f.readline().strip().split()
                if len(dims)!=2:raise ValueError('GCT #1.2 requires two dimensions')
                n_genes,n_samples=map(int,dims)
                header=f.readline().rstrip('\r\n').split('\t')
                skiprows=2
                expression_format='GCT #1.2'
            elif first.split('\t')[:2]==['Name','Description']:
                header=first.split('\t')
                n_genes=None
                n_samples=len(header)-2
                skiprows=0
                expression_format='Plain TSV with Name/Description header'
            else:
                raise ValueError('Unsupported expression header: '+repr(first[:200]))
        samples=header[2:]
        if n_samples<1 or len(samples)!=n_samples or len(set(samples))!=n_samples:raise ValueError('Expression sample header/dimensions mismatch or duplicate samples')
        report['expression_format']=expression_format
        gct_ids=[];descriptions=[];rows=0;nonfinite=0;negative=0
        for chunk in pd.read_csv(gct,sep='\t',skiprows=skiprows,chunksize=1000,low_memory=False):
            if len(chunk.columns)!=n_samples+2:raise ValueError('GCT column count mismatch')
            a=chunk.iloc[:,2:].apply(pd.to_numeric,errors='raise').to_numpy(float)
            nonfinite+=int((~np.isfinite(a)).sum());negative+=int((a<0).sum());rows+=len(chunk)
            gct_ids.extend(chunk.iloc[:,0].astype(str));descriptions.extend(chunk.iloc[:,1].fillna('').astype(str))
        if n_genes is not None and rows!=n_genes:raise ValueError('GCT gene row count does not match dimensions')
        if nonfinite or negative:report['errors'].append(f'GCT has {nonfinite} nonfinite and {negative} negative values')
        report['expression']={'shape_genes_samples':[rows,n_samples],'sample_examples':samples[:5],'gene_examples':gct_ids[:5],'description_examples':descriptions[:5],'duplicate_gene_ids':len(gct_ids)-len(set(gct_ids))}
        dep_ids=[];dep_columns=None;dep_nonfinite=0
        for chunk in pd.read_csv(bench/required[1],index_col=0,chunksize=100,low_memory=False):
            dep_columns=chunk.columns.tolist();dep_ids.extend(chunk.index.astype(str));a=chunk.apply(pd.to_numeric,errors='raise').to_numpy(float);dep_nonfinite+=int((~np.isfinite(a)).sum())
        if len(dep_ids)!=len(set(dep_ids)):report['errors'].append('Duplicate dependency sample IDs')
        report['dependency']={'samples':len(dep_ids),'genes':len(dep_columns or []),'sample_examples':dep_ids[:5],'gene_examples':(dep_columns or [])[:5],'nonfinite_values':dep_nonfinite}
        if dep_nonfinite:report['warnings'].append('Dependency has missing/nonfinite labels; evaluation must explicitly exclude them')
        # Find explicit sample mapping in deposited metadata; no name guessing.
        mapping={s:s for s in samples if s in set(cls['cell-line'])}
        metadata_path=bench/'depmap/raw/depmap-metadata.csv'
        if metadata_path.is_file():
            md=pd.read_csv(metadata_path,dtype=str).fillna('')
            report['schemas']['depmap/raw/depmap-metadata.csv']={'rows':len(md),'columns':md.columns.tolist()}
            target=set(cls['cell-line']);source=set(samples)
            for a_col in md.columns:
                if not set(md[a_col])&source:continue
                for b_col in md.columns:
                    if a_col==b_col or not set(md[b_col])&target:continue
                    for a,b in zip(md[a_col],md[b_col]):
                        if a in source and b in target:
                            if a in mapping and mapping[a]!=b:raise ValueError('Conflicting explicit sample mapping for '+a)
                            mapping[a]=b
        report['sample_mapping']={'expression_samples_mapped':len(mapping),'split_samples':len(cls),'metadata_present':metadata_path.is_file()}
        pd.DataFrame(sorted(mapping.items()),columns=['expression_sample','cell-line']).to_csv(out/'sample_mapping.csv',index=False)
        if len(set(mapping.values()))!=len(mapping):report['errors'].append('Multiple expression columns map to the same split sample; aggregation rule unresolved')
        common=set(mapping.values())&set(dep_ids)&set(cls['cell-line'])
        report['folds']={}
        for fold in ['fold-'+str(i) for i in range(5)]:
            train=set(cls.loc[cls[fold]=='train','cell-line']);val=set(cls.loc[cls[fold]=='val','cell-line'])
            report['folds'][fold]={'train_in_split':len(train),'val_in_split':len(val),'train_with_expression_and_dependency':len(train&common),'val_with_expression_and_dependency':len(val&common)}
            if len(train&common)<512:report['errors'].append(f'{fold}: fewer than 512 mapped training samples; requested PCA dimension unsupported')
            if not val&common:report['errors'].append(f'{fold}: no mapped validation samples')
            missing=(train|val)-common
            if missing:report['warnings'].append(f'{fold}: {len(missing)} split samples unavailable in expression/dependency intersection')
        pathways={}
        with (bench/required[4]).open(encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():continue
                parts=line.rstrip('\r\n').split('\t')
                if len(parts)<3 or parts[0] in pathways:raise ValueError('Malformed/duplicate GMT pathway')
                pathways[parts[0]]=set(parts[2:])
        if len(pathways)!=50:report['errors'].append(f'Expected 50 Hallmark pathways; found {len(pathways)}')
        dep_genes={c.split(' (')[0].strip() for c in dep_columns}
        split_genes=set(genes['gene']);symbols=set(descriptions);ids=set(gct_ids)
        report['gene_overlap']={'split_dependency':len(split_genes&dep_genes),'split_gct_ids':len(split_genes&ids),'split_gct_descriptions':len(split_genes&symbols),'hallmark_gct_ids':len(set.union(*pathways.values())&ids),'hallmark_gct_descriptions':len(set.union(*pathways.values())&symbols)}
        gm=bench/'depmap/raw/gene-mapping.csv'
        if gm.is_file():
            table=pd.read_csv(gm,dtype=str).fillna('');report['schemas']['depmap/raw/gene-mapping.csv']={'rows':len(table),'columns':table.columns.tolist(),'examples':table.head(3).to_dict('records')}
        if not split_genes&dep_genes:report['errors'].append('No split-gene/dependency-gene overlap')
        if not split_genes&(ids|symbols):
            if not gm.is_file():report['errors'].append('Expression gene IDs need mapping; gene-mapping.csv missing')
            else:report['warnings'].append('Expression gene mapping requires interpretation of gene-mapping.csv; schema/examples included')
        report['hallmark_pathways']=len(pathways)
        report['protocol_pending']=['Verify gene mapping and duplicate aggregation','Match upstream DepMap classifier and fold protocol','Use MSigDB-specific expression data and held-out evaluation; do not train and score on identical genes']
    except Exception as e:report['errors'].append(type(e).__name__+': '+str(e))
    return finish()

if __name__=='__main__':
    result=check()
    sys.exit(0 if result['input_checks_passed'] else 1)
