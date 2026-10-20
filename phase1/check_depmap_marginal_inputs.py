"""Read-only marginal-essentiality input audit. Save in phase1; no training."""
from pathlib import Path
import hashlib, json, platform
import h5py
import numpy as np
import pandas as pd

BASE = Path(r"D:\data\phase1\benchmarks\depmap")
OUT = Path(__file__).resolve().parent / "results/depmap_marginal_preflight"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def vector(node):
    if isinstance(node, h5py.Group):
        if not {"codes", "categories"}.issubset(node):
            raise ValueError("Unsupported observation encoding")
        codes = node["codes"][()]
        if (codes < 0).any():
            raise ValueError("Missing categorical observations")
        return vector(node["categories"])[codes]
    a = node[()]
    if a.dtype.kind in "SO":
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])
    return a

def inspect(path):
    record = dict(file=path.name, bytes=path.stat().st_size, sha256=digest(path))
    with h5py.File(path, "r") as f:
        record["obs_columns"] = list(f["obs"])
        genes, labels = vector(f["obs/gene"]), vector(f["obs/score"])
        if genes.ndim != 1 or labels.ndim != 1 or len(genes) != len(labels):
            raise ValueError("Invalid observation shapes")
        if len(set(genes)) != len(genes) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Duplicate genes or nonbinary labels")
        node = f["X"]
        if isinstance(node, h5py.Dataset):
            shape = node.shape
            finite = all(np.isfinite(node[start:start+256]).all() for start in range(0, shape[0], 256))
            encoding, dtype = "dense", str(node.dtype)
        else:
            encoding = node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            if encoding not in {"csr_matrix", "csc_matrix"}:
                raise ValueError("Unsupported matrix encoding")
            shape = tuple(node.attrs["shape"])
            data = node["data"]
            finite = all(np.isfinite(data[start:start+1000000]).all() for start in range(0, len(data), 1000000))
            dtype = str(data.dtype)
            indices, ptr = node["indices"][()], node["indptr"][()]
            axis, bound = (shape[0], shape[1]) if encoding == "csr_matrix" else (shape[1], shape[0])
            if len(ptr) != axis+1 or ptr[0] != 0 or ptr[-1] != len(data) or len(indices) != len(data) or (np.diff(ptr)<0).any() or (indices<0).any() or (indices>=bound).any():
                raise ValueError("Invalid sparse matrix structure")
        if len(shape) != 2 or shape[0] != len(genes) or shape[1] == 0 or not finite:
            raise ValueError("Invalid matrix shape or nonfinite features")
        record.update(shape=list(shape), encoding=encoding, dtype=dtype, finite_features=bool(finite), genes=len(genes), positives=int((labels==1).sum()), negatives=int((labels==0).sum()))
    return record, genes, labels.astype(int)

def run(base=BASE, out=OUT):
    base, out = Path(base), Path(out)
    splitpath = base / "misc/split-genes-lt5gt70.csv"
    paths = sorted((base / "gene-embs").glob("*-mean-lt5gt70-bin.h5ad"))
    pca = base / "gene-embs/pcloading-lt5gt70-bin.h5ad"
    report = dict(scope="Local input contract inspection only; no RF training or checkpoint inference", versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, h5py=h5py.__version__), artifacts=[], errors=[])
    if not splitpath.is_file() or not pca.is_file():
        raise FileNotFoundError("Required split or PCA artifact missing")
    split = pd.read_csv(splitpath, dtype=str)
    if "gene" not in split or not split.gene.is_unique or split.isna().any().any():
        raise ValueError("Invalid split table")
    columns = ["fold-"+str(i) for i in range(5)]
    if not all(c in split and set(split[c]) == {"train", "val"} for c in columns):
        raise ValueError("Expected five train/val gene folds")
    if not ((split[columns]=="val").sum(axis=1)==1).all():
        raise ValueError("Each split gene must be validated once")
    reference, pgenes, plabels = inspect(pca)
    if reference["shape"][1] != 15 or set(pgenes) != set(split.gene):
        raise ValueError("PCA/split contract mismatch")
    labelmap = dict(zip(pgenes, plabels))
    split = split.set_index("gene")
    report["split_sha256"] = digest(splitpath)
    report["split_genes"] = len(split)
    gene_records = []
    if not paths:
        report["errors"].append(dict(file="gene-embs", error="No model marginal artifacts found"))
    for path in [pca]+paths:
        try:
            record, genes, labels = inspect(path)
            unknown = sorted(set(genes)-set(split.index))
            mismatch = [g for g,y in zip(genes,labels) if g in labelmap and labelmap[g]!=y]
            record.update(genes_outside_split=unknown, label_mismatches=mismatch, missing_split_genes=len(set(split.index)-set(genes)), folds=[])
            if unknown or mismatch:
                raise ValueError("Genes outside split or labels disagree with PCA reference")
            aligned = split.loc[genes]
            for c in columns:
                train, val = (aligned[c]=="train").to_numpy(), (aligned[c]=="val").to_numpy()
                record["folds"].append(dict(fold=c, train=int(train.sum()), val=int(val.sum()), train_positives=int(labels[train].sum()), val_positives=int(labels[val].sum()), train_classes=np.unique(labels[train]).tolist(), val_classes=np.unique(labels[val]).tolist()))
            report["artifacts"].append(record)
            gene_records.extend(dict(artifact=path.name,gene=g,score=int(y)) for g,y in zip(genes,labels))
            print(path.name, record["shape"], "missing split genes:",record["missing_split_genes"],flush=True)
        except Exception as exc:
            report["errors"].append(dict(file=path.name,error=str(exc)))
            print("FAILED:",path.name,str(exc),flush=True)
    sets = [set(x["gene"] for x in gene_records if x["artifact"]==r["file"]) for r in report["artifacts"]]
    common = sorted(set.intersection(*sets)) if sets else []
    report["common_genes"] = len(common)
    report["common_fold_counts"] = {c:split.loc[common,c].value_counts().to_dict() for c in columns}
    report["label_check_limit"] = "Agreement with deposited PCA labels only; raw dependency labels not reconstructed by this checker"
    report["status"] = "PASS" if not report["errors"] else "INCOMPLETE"
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(gene_records).to_csv(out/"gene_labels.csv",index=False)
    pd.DataFrame({"gene":common}).to_csv(out/"common_genes.csv",index=False)
    for r in report["artifacts"]:
        if digest(base/"gene-embs"/r["file"]) != r["sha256"]:
            raise RuntimeError("Input changed during audit")
    target=out/"report.json"
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    assert json.loads(target.read_text(encoding="utf-8"))==report
    print("Status:",report["status"],"common genes:",len(common),"errors:",len(report["errors"]))
    print(target.resolve())
    return report

if globals().get("__name__")=="__main__":
    run()
