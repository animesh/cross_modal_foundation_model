"""Prepare 308 RNA-only control profiles and reserve 1024 disjoint evaluation cells.
Independent experiment split, not the paper split; original pretraining overlap unknown.
"""
from pathlib import Path
import json,gzip,shutil
import numpy as np
import h5py
from scipy import sparse
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/rna_control_inputs'

def text(node):
    if isinstance(node,h5py.Group):
        codes=node['codes'][()]
        assert (codes>=0).all(),'Missing feature IDs'
        return text(node['categories'])[codes]
    return np.array([v.decode() if isinstance(v,bytes) else str(v) for v in node[()]])

def prepare(source,out=OUT):
    out=Path(out);assert not out.exists(),'RNA input output already exists'
    out.mkdir(parents=True)
    actual=Path(source)
    with actual.open('rb') as f:magic=f.read(2)
    temporary=None
    if magic==bytes([31,139]):
        temporary=out/'source_uncompressed.h5ad'
        with gzip.open(actual,'rb') as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst,8388608)
        actual=temporary
    vocab=json.loads((ROOT/'models/tahoe_original/70m-model/vocab.json').read_text())
    with h5py.File(actual,'r') as f:
        features=text(f['var/feature_id'])
        mapped=[(i,int(vocab[g.split('.')[0]])) for i,g in enumerate(features) if g.split('.')[0] in vocab and int(vocab[g.split('.')[0]])>0]
        assert mapped,'No feature IDs match vocabulary'
        ids=np.array(sorted({t for _,t in mapped}),dtype=np.int64);lookup={int(t):i for i,t in enumerate(ids)}
        X=f['X']
        if isinstance(X,h5py.Dataset):shape=X.shape;encoding='dense'
        else:
            shape=tuple(X.attrs['shape']);encoding=X.attrs['encoding-type']
            if isinstance(encoding,bytes):encoding=encoding.decode()
            assert encoding=='csr_matrix','Unsupported source matrix encoding: '+str(encoding)
        assert shape[1]==len(features)
        mapping=sparse.csr_matrix((np.ones(len(mapped)),([i for i,t in mapped],[lookup[t] for i,t in mapped])),shape=(shape[1],len(ids)))
        rng=np.random.default_rng(20261019);order=rng.permutation(shape[0])
        accepted=[];profiles=[]
        for index in order:
            index=int(index)
            if encoding=='dense':row=sparse.csr_matrix(X[index:index+1])
            else:
                lo,hi=map(int,X['indptr'][index:index+2])
                row=sparse.csr_matrix((X['data'][lo:hi],X['indices'][lo:hi],np.array([0,hi-lo])),shape=(1,shape[1]))
            assert np.isfinite(row.data).all() and (row.data>=0).all(),'Invalid RNA values'
            assert np.allclose(row.data,np.rint(row.data),rtol=0,atol=1e-5),'RNA X not integer-like counts'
            values=(row@mapping).toarray().ravel().astype(np.float32)
            if np.count_nonzero(values)<1023:continue
            accepted.append(index)
            if len(profiles)<308:profiles.append(values)
            if len(accepted)==1332:break
        assert len(accepted)==1332,'Insufficient eligible cells'
    train=np.array(accepted[:308]);evaluation=np.array(accepted[308:])
    assert not set(train)&set(evaluation)
    np.savez_compressed(out/'profiles.npz',gene_ids=ids,lfq=np.stack(profiles),samples=np.array(['rna_row_'+str(i) for i in train]))
    np.savez_compressed(out/'cell_split.npz',train_rows=train,evaluation_rows=evaluation)
    manifest={'source':str(Path(source).resolve()),'source_bytes':Path(source).stat().st_size,'source_mtime_ns':Path(source).stat().st_mtime_ns,'training_cells':308,'evaluation_cells':1024,'seed':20261019,'eligibility':'At least1023 positive vocabulary-mapped genes; sampled without replacement','limits':'Evaluation cells disjoint from this continued-training control, not necessarily original model pretraining; no study/donor split established','feature_mapping':'Version-stripped Ensembl IDs; sum duplicate IDs','matrix_encoding':encoding,'mapped_genes':len(ids)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if temporary:temporary.unlink()
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    roots=[Path('F:/data'),Path('D:/data'),Path('L:/data')]
    source=next(r/'phase1/benchmarks/msigdb/cellxgene_ds_data/rep_1.h5ad.gz' for r in roots if (r/'phase1/benchmarks/msigdb/cellxgene_ds_data/rep_1.h5ad.gz').is_file())
    prepare(source)
