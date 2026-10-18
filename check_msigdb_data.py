"""Read-only MSigDB input/runtime audit. No training or input modification.
Run from phase1: python check_msigdb_data.py
"""
from pathlib import Path
import sys,json,importlib
import numpy as np
BASE=Path(r'D:\data\phase1\benchmarks\msigdb')
OUT=Path(__file__).resolve().parent/'results/msigdb_preflight'

def json_numpy(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def check(base=BASE,out=OUT):
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'root':str(base),'python':sys.version,'executable':sys.executable,'files':[],'npz':[],'gmt':[],'h5ad':[],'runtime':{},'errors':[]}
    if not base.is_dir():report['errors'].append('Benchmark directory missing')
    for path in sorted(base.rglob('*')):
        if not path.is_file():continue
        rel=str(path.relative_to(base));report['files'].append({'path':rel,'bytes':path.stat().st_size})
        try:
            if path.suffix.lower()=='.npz':
                entry={'path':rel,'arrays':{}}
                with np.load(path,allow_pickle=False) as archive:
                    for key in archive.files:
                        try:
                            a=archive[key];info={'shape':list(a.shape),'dtype':str(a.dtype)}
                            if a.dtype.kind in 'fiu':info['nonfinite']=int((~np.isfinite(a)).sum())
                            if a.ndim==1 and a.dtype.kind in 'SU':info['examples']=a[:3].astype(str).tolist();info['unique']=len(np.unique(a))
                            entry['arrays'][key]=info
                        except ValueError as e:entry['arrays'][key]={'inspection_blocked':str(e),'note':'Object arrays not unpickled by audit'}
                report['npz'].append(entry)
            elif path.suffix.lower()=='.gmt':
                pairs=[];sizes={}
                with path.open(encoding='utf-8-sig') as f:
                    for line in f:
                        if not line.strip():continue
                        parts=line.rstrip('\r\n').split('\t')
                        if len(parts)<3:raise ValueError('Malformed GMT row')
                        sizes[parts[0]]=sizes.get(parts[0],0)+len(parts[2:]);pairs.extend((parts[0],g) for g in parts[2:])
                report['gmt'].append({'path':rel,'signatures':len(sizes),'genes':len({g for _,g in pairs}),'duplicate_memberships':len(pairs)-len(set(pairs)),'signature_examples':list(sizes)[:3]})
            elif '.h5ad' in path.name.lower():
                import h5py
                with h5py.File(path,'r') as f:
                    x=f['X'];shape=list(x.shape) if isinstance(x,h5py.Dataset) else list(x.attrs.get('shape',[]))
                    obs=f['obs'];var=f['var']
                    entry={'path':rel,'X_shape':shape,'obs_columns':list(obs.keys()),'var_columns':list(var.keys()),'obsm':{}}
                    for name,node in f.get('obsm',{}).items():
                        entry['obsm'][name]=list(node.shape) if isinstance(node,h5py.Dataset) else {'encoding':str(node.attrs.get('encoding-type',''))}
                    index=obs.attrs.get('_index','_index')
                    if isinstance(index,bytes):index=index.decode()
                    if index in obs and isinstance(obs[index],h5py.Dataset):
                        a=obs[index][()];entry['gene_index_examples']=[v.decode() if isinstance(v,bytes) else str(v) for v in a[:3]];entry['gene_index_unique']=len(np.unique(a))
                    report['h5ad'].append(entry)
        except Exception as e:report['errors'].append(rel+': '+type(e).__name__+': '+str(e))
    for name in ['numpy','pandas','sklearn','h5py','torch','pytorch_lightning','anndata','omegaconf']:
        try:
            module=importlib.import_module(name);report['runtime'][name]={'version':str(getattr(module,'__version__','unknown'))}
            if name=='torch':
                report['runtime'][name].update(cuda_available=module.cuda.is_available(),cuda_version=module.version.cuda,gpu_count=module.cuda.device_count(),gpu_names=[module.cuda.get_device_name(i) for i in range(module.cuda.device_count())])
        except Exception as e:report['runtime'][name]={'unavailable':type(e).__name__+': '+str(e)}
    report['scope']='Input/schema/runtime inventory only; not training validation. Object arrays deliberately not unpickled.'
    (out/'report.json').write_text(json.dumps(report,indent=2,default=json_numpy),encoding='utf-8')
    print('Files:',len(report['files']),'NPZ:',len(report['npz']),'GMT:',len(report['gmt']),'H5AD:',len(report['h5ad']))
    print('Errors:',report['errors']);print('Runtime:',json.dumps(report['runtime'],indent=2,default=json_numpy));print('Report:',out/'report.json')
    return report

if __name__=='__main__':check()
