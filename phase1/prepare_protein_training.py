"""Prepare measured protein profiles. Small-corpus experiment, not paper reproduction.
Uses formal metadata to exclude unmatched LFQ columns. Does not train or change weights.
"""
from pathlib import Path
import csv,json,collections,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def mapping_rows(contract):
    lookup=collections.defaultdict(set)
    for feature in contract['expression_features']:
        name=feature['name'];gid=feature['ensembl'].split('.')[0]
        if name.endswith('_'+gid):name=name[:-len(gid)-1]
        if feature['in_vocab']:lookup[name].add(gid)
    mapped=[]
    for row in contract['protein_groups']:
        symbols={s.strip() for s in row['genes'].split(';') if s.strip()}
        hits=[lookup[s] for s in symbols]
        gene=next(iter(set.union(*hits))) if symbols and all(len(h)==1 for h in hits) and len(set.union(*hits))==1 else None
        mapped.append(gene)
    return mapped

def run():
    csv.field_size_limit(2147483647)
    contract=json.loads((ROOT/'results/protein_mapping_contract.json').read_text())
    pg=Path(contract['protein_source']);meta=pg.parent/'Metadata-Table.xlsx'
    vocab=json.loads((ROOT/'models/tahoe_original/70m-model/vocab.json').read_text())
    annotations=pd.read_excel(meta,engine='openpyxl')
    def key(value):
        text=str(value).strip().lower()
        return text[:-4] if text.endswith('.raw') else text
    keys=[key(v) for v in annotations['Raw Mass Spectrometry File']]
    assert len(keys)==len(set(keys)), 'Duplicate metadata sample keys'
    columns=contract['lfq_columns'];lookup={key(c[14:]):c for c in columns}
    assert len(lookup)==len(columns),'Duplicate LFQ sample keys'
    assert set(keys)<=set(lookup),'Metadata sample missing LFQ column'
    selected=[lookup[k] for k in keys]
    mapped=mapping_rows(contract);rows=collections.defaultdict(lambda:np.zeros(len(selected),dtype=np.float64))
    position=0
    with pg.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream,delimiter='	')
        assert [c for c in reader.fieldnames if c.startswith('LFQ intensity ')]==columns,'LFQ contract changed'
        for row in reader:
            if any(row.get(k,'')=='+' for k in ['Reverse','Potential contaminant','Only identified by site']):continue
            assert position<len(mapped),'More protein groups than contract'
            recorded=contract['protein_groups'][position]
            assert row['Gene names']==recorded['genes'] and row.get('Protein IDs','')==recorded['proteins'],'Protein-group contract changed'
            gene=mapped[position];position+=1
            if gene is None:continue
            assert gene in vocab,'Mapping absent from current vocabulary'
            values=np.array([float(row[c]) if row[c] else 0 for c in selected])
            assert np.isfinite(values).all() and (values>=0).all(),'Invalid LFQ values'
            rows[int(vocab[gene])]+=values
    assert position==len(mapped),'Incomplete protein-group read'
    ids=np.array(sorted(rows),dtype=np.int64)
    lfq=np.stack([rows[int(i)] for i in ids],axis=1).astype(np.float32)
    assert np.isfinite(lfq).all() and (np.count_nonzero(lfq,axis=1)>=2).all(),'Invalid mapped sample'
    out=ROOT/'results/protein_training_inputs';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'profiles.npz',gene_ids=ids,lfq=lfq,samples=np.array(keys,dtype=str))
    annotations.to_csv(out/'sample_metadata.csv',index=False)
    with np.load(out/'profiles.npz',allow_pickle=False) as z:
        assert np.array_equal(z['lfq'],lfq) and np.array_equal(z['gene_ids'],ids) and list(z['samples'])==keys
    report={'scope':'Real PXD014791 small-corpus training input; not full paper reproduction. Do not reuse this dataset as held-out protein evaluation.','samples':len(keys),'mapped_genes':len(ids),'mapped_groups':sum(g is not None for g in mapped),'excluded_groups':sum(g is None for g in mapped),'excluded_LFQ_columns':[c for c in columns if c not in selected],'aggregation':'Sum same-token LFQ groups; zero means unmeasured, not a training target; no imputation','measured_genes_per_sample':np.count_nonzero(lfq,axis=1).tolist()}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['samples','mapped_genes','mapped_groups','excluded_groups']},indent=2))
    print(out.resolve())

if __name__=='__main__':
    run()
