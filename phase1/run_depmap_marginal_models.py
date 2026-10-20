"""Five-fold RF evaluation of deposited marginal gene embeddings; no input changes.
Save in phase1 beside check_depmap_marginal_inputs.py.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import sparse
from check_depmap_marginal_inputs import BASE, digest, inspect

OUT = Path(__file__).resolve().parent / "results/depmap_marginal_models"

def matrix(path):
    with h5py.File(path, "r") as f:
        node = f["X"]
        if isinstance(node, h5py.Dataset):
            return node[()]
        encoding = node.attrs["encoding-type"]
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        cls = {"csr_matrix": sparse.csr_matrix, "csc_matrix": sparse.csc_matrix}[encoding]
        return cls((node["data"][()], node["indices"][()], node["indptr"][()]), shape=tuple(node.attrs["shape"]))

def run(base=BASE, out=OUT):
    base, out = Path(base), Path(out)
    paths = [base/"gene-embs/pcloading-lt5gt70-bin.h5ad"] + sorted((base/"gene-embs").glob("*-mean-lt5gt70-bin.h5ad"))
    if len(paths)<2:
        raise ValueError("No model artifacts found")
    splitpath=base/"misc/split-genes-lt5gt70.csv"
    split=pd.read_csv(splitpath,dtype=str)
    cols=["fold-"+str(i) for i in range(5)]
    if not split.gene.is_unique or split.isna().any().any() or not all(set(split[c])=={"train","val"} for c in cols):
        raise ValueError("Invalid split")
    if not ((split[cols]=="val").sum(axis=1)==1).all():
        raise ValueError("Each gene must be validated once")
    split=split.set_index("gene")
    metadata=[]
    for path in paths:
        record, genes, labels=inspect(path)
        if not set(genes).issubset(split.index):
            raise ValueError("Artifact contains genes outside split: "+path.name)
        metadata.append((path,record,genes,labels))
    _,_,pgenes,plabels=metadata[0]
    if set(pgenes)!=set(split.index):
        raise ValueError("PCA gene universe differs from split")
    reference=dict(zip(pgenes,plabels))
    for path,record,genes,labels in metadata:
        if not all(reference[g]==int(y) for g,y in zip(genes,labels)):
            raise ValueError("Label disagreement: "+path.name)
    common=sorted(set.intersection(*(set(item[2]) for item in metadata)))
    if not common:
        raise ValueError("Empty common gene universe")
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"gene":common}).to_csv(out/"common_genes.csv",index=False)
    rows=[]; predictions=[]
    for path,record,genes,labels in metadata:
        X=matrix(path)
        model=path.stem
        positions={g:i for i,g in enumerate(genes)}
        for scenario in ("native","common"):
            idx=np.arange(len(genes)) if scenario=="native" else np.array([positions[g] for g in common])
            selected_genes=genes[idx]; y=labels[idx]; features=X[idx]
            aligned=split.loc[selected_genes]
            seen=[]
            for fold,c in enumerate(cols):
                train=(aligned[c]=="train").to_numpy(); val=(aligned[c]=="val").to_numpy()
                if len(np.unique(y[train]))!=2 or len(np.unique(y[val]))!=2:
                    raise ValueError("Both classes required: "+model+" "+scenario+" "+c)
                rf=RandomForestClassifier(n_jobs=8,random_state=42)
                rf.fit(features[train],y[train])
                if not np.array_equal(rf.classes_,[0,1]):
                    raise ValueError("Unexpected RF class order")
                p=rf.predict_proba(features[val])[:,1]
                if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
                    raise ValueError("Invalid predictions")
                rows.append(dict(model=model,scenario=scenario,fold=fold,n_train=int(train.sum()),n_val=int(val.sum()),n_positive=int(y[val].sum()),prevalence=float(y[val].mean()),auprc=float(average_precision_score(y[val],p)),auroc=float(roc_auc_score(y[val],p))))
                predictions.append(pd.DataFrame(dict(model=model,scenario=scenario,fold=fold,gene=selected_genes[val],score=y[val],probability=p)))
                seen.extend(selected_genes[val])
                print(model,scenario,c,"AUPRC",format(rows[-1]["auprc"],".6f"),flush=True)
            if len(seen)!=len(selected_genes) or len(set(seen))!=len(seen):
                raise ValueError("Incomplete validation coverage")
        if digest(path)!=record["sha256"]:
            raise RuntimeError("Input changed during run")
        pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
        pd.concat(predictions,ignore_index=True).to_csv(out/"predictions.csv",index=False)
    metrics=pd.DataFrame(rows)
    summary=metrics.groupby(["scenario","model"]).agg(mean_auprc=("auprc","mean"),sd_auprc=("auprc","std"),mean_auroc=("auroc","mean"),folds=("fold","count"),validation_genes=("n_val","sum")).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    # Independently recompute every fold metric from the saved CSV.
    saved=pd.read_csv(out/"predictions.csv")
    if saved.duplicated(["model","scenario","gene"]).any():
        raise ValueError("Duplicate validation predictions")
    for row in rows:
        subset=saved[(saved.model==row["model"])&(saved.scenario==row["scenario"])&(saved.fold==row["fold"])]
        assert len(subset)==row["n_val"]
        assert np.isclose(average_precision_score(subset.score,subset.probability),row["auprc"],rtol=0,atol=1e-12)
        assert np.isclose(roc_auc_score(subset.score,subset.probability),row["auroc"],rtol=0,atol=1e-12)
    notes=dict(scope="Deposited marginal-essentiality embeddings; not verified target-paper checkpoint comparison",RF="sklearn defaults, n_jobs=8, random_state=42",scenarios={"native":"Train and evaluate each model on its available genes","common":"Train and evaluate every model on the identical intersection; preserve original folds"},common_genes=len(common),artifacts=[m[1] for m in metadata],split_sha256=digest(splitpath),labels="Checked against deposited PCA labels; raw dependency labels not reconstructed here",versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,h5py=h5py.__version__),checks="Finite matrices; labels; split coverage; hashes; saved predictions and all fold metrics recomputed",limitations="Native coverage/prevalence differs; common universe is a secondary sensitivity analysis; no missing embeddings imputed; no model checkpoint loaded")
    (out/"assumptions.json").write_text(json.dumps(notes,indent=2),encoding="utf-8")
    print(summary.to_string(index=False))
    return summary

if __name__=="__main__":
    run()
