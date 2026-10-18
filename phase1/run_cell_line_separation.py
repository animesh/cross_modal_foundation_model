"""Read-only upstream-style tissue enrichment; save in phase1 and run with Python.
No UMAP computation, input rewriting, model inference, or feature normalization.
"""
from pathlib import Path
import io, json, hashlib, platform
import numpy as np
import pandas as pd
import h5py
import sklearn
from sklearn.neighbors import kneighbors_graph
BASE = Path(r"D:\data\phase1\benchmarks\depmap\cell-embs")
OUT = Path(__file__).resolve().parent / "results/cell_line_separation"

def read_strings(node):
    if isinstance(node, h5py.Group):
        codes = node["codes"][()]
        if (codes < 0).any():
            raise ValueError("Missing tissue labels or identifiers")
        return read_strings(node["categories"])[codes]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in node[()]])

def read_embedding(data):
    with h5py.File(io.BytesIO(data), "r") as f:
        X = f["X"][()]
        obs = f["obs"]
        key = obs.attrs["_index"]
        if isinstance(key, bytes):
            key = key.decode()
        names = read_strings(obs[key])
        ids = read_strings(obs["ModelID"])
        labels = read_strings(obs["OncotreeLineage"])
    if X.ndim != 2 or len(X) != len(names) or len(ids) != len(X) or len(labels) != len(X):
        raise ValueError("Embedding/metadata shape mismatch")
    if not np.isfinite(X).all() or len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Nonfinite embeddings or duplicate identifiers")
    return X, names, ids, labels

def score_embedding(X, labels, k=20):
    if len(X) <= k:
        raise ValueError("Too few samples for k=20")
    graph = kneighbors_graph(X, n_neighbors=k).tocoo()
    codes = pd.Categorical(labels).codes
    same = codes[graph.row] == codes[graph.col]
    _, counts = np.unique(codes, return_counts=True)
    baseline = float(((counts / counts.sum()) ** 2).sum())
    fraction = float(same.mean())
    return fraction / baseline, fraction, baseline, graph, same

def run(base=BASE, out=OUT):
    base, out = Path(base), Path(out)
    paths = sorted(base.glob("*.h5ad"))
    if len(paths) != 15:
        raise ValueError(f"Expected 15 H5AD files, found {len(paths)}")
    loaded = [(p, p.read_bytes()) for p in paths]
    embeddings = [(p, data, read_embedding(data)) for p, data in loaded]
    reference = embeddings[0][2]
    for p, data, (X, names, ids, labels) in embeddings:
        if not all(np.array_equal(a, b) for a, b in zip((ids, labels), reference[2:])):
            raise ValueError(f"Sample order or tissue labels differ: {p.name}")
    out.mkdir(parents=True, exist_ok=True)
    rows, cells, edges, inventory = [], [], [], []
    for p, data, (X, names, ids, labels) in embeddings:
        score, fraction, baseline, graph, same = score_embedding(X, labels)
        counts = np.bincount(graph.row, weights=same.astype(int), minlength=len(X)).astype(int)
        assert graph.nnz == len(X) * 20 and not (graph.row == graph.col).any()
        assert np.all(np.bincount(graph.row, minlength=len(X)) == 20)
        assert abs(counts.mean() / 20 / baseline - score) < 1e-12
        rows.append(dict(embedding=p.stem, samples=len(X), dimensions=X.shape[1], tissues=len(set(labels)), k=20, same_tissue_fraction=fraction, frequency_baseline=baseline, enrichment=score))
        cells.append(pd.DataFrame(dict(embedding=p.stem, sample=names, ModelID=ids, tissue=labels, same_tissue_neighbors=counts, same_tissue_fraction=counts / 20)))
        edges.append(pd.DataFrame(dict(embedding=p.stem, query=ids[graph.row], neighbor=ids[graph.col], same_tissue=same)))
        inventory.append(dict(file=p.name, sha256=hashlib.sha256(data).hexdigest(), dtype=str(X.dtype), shape=list(X.shape)))
        print(f"{p.stem}: {score:.6f}", flush=True)
    summary = pd.DataFrame(rows).sort_values("enrichment", ascending=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(out / "per_cell.csv", index=False)
    pd.concat(edges, ignore_index=True).to_csv(out / "neighbors.csv", index=False)
    pd.Series(reference[3]).value_counts().rename_axis("tissue").reset_index(name="samples").to_csv(out / "tissue_counts.csv", index=False)
    notes = dict(metric="Upstream compute_lisi_scores: same-label fraction over directed Euclidean k=20 neighbors divided by sum of squared tissue frequencies; not conventional inverse-Simpson LISI", normalization="None; original X and dtype retained", scope="Deposited embeddings only; checkpoint provenance and exact target-paper equivalence unverified; no UMAP or training", python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, inputs=inventory)
    (out / "assumptions.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    for p, data in loaded:
        assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(data).digest(), "Input changed"
    return summary

if __name__ == "__main__":
    run()
