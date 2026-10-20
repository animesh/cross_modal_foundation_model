"""Read-only provenance evidence collection. Run from the repository root.
Uses standard Python plus h5py for H5AD metadata; never downloads weights,
loads pickle checkpoints, changes inputs, or infers training modality from names.
"""
from pathlib import Path
import datetime, hashlib, json, os, re, subprocess, urllib.request, urllib.error

REPO = Path(__file__).resolve().parent
UPSTREAM = REPO.parent / "tahoe-x1-upstream"
BENCHMARKS = Path(r"D:\data\phase1\benchmarks")
OUT = REPO / "results/checkpoint_provenance"
HF_MODELS = ["mburq/Trove-1-70m"]
GITHUB_REPOS = ["tesorai/cross_modal_foundation_model", "tahoebio/tahoe-x1"]
PATTERN = re.compile(r"trove|mosaicfm|tx1|tahoe|checkpoint|pretrain|proteom|alternate|revision|sha256|manifest", re.I)
MAX_TEXT_BYTES = 2 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, token=None):
    headers = {"User-Agent": "checkpoint-provenance-audit/1.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(MAX_TEXT_BYTES + 1)
    if len(data) > MAX_TEXT_BYTES:
        raise ValueError("Response exceeds metadata size limit: " + url)
    return data


def get_json(url, token=None):
    return json.loads(fetch(url, token))


def collect_hf(model):
    from urllib.parse import quote
    token = os.environ.get("HF_TOKEN")
    base = "https://huggingface.co"
    info = get_json(base + "/api/models/" + model + "?blobs=true", token)
    revision = info.get("sha")
    if not revision:
        raise ValueError("No resolved model revision: " + model)
    # Re-query the immutable revision to avoid mixing mutable main snapshots.
    pinned = get_json(base + "/api/models/" + model + "/revision/" + revision + "?blobs=true", token)
    if pinned.get("sha") != revision:
        raise ValueError("Revision mismatch: " + model)
    files, texts, failures = [], [], []
    for item in pinned.get("siblings", []):
        name = item["rfilename"]
        lfs = item.get("lfs") or {}
        files.append(dict(path=name, size=item.get("size"), git_blob_id=item.get("blobId"),
                          published_lfs_sha256=lfs.get("sha256"),
                          hash_status="published metadata only; weights not downloaded"))
        if name.endswith((".json", ".yaml", ".yml", ".md")):
            size = item.get("size")
            if size is None or size > MAX_TEXT_BYTES:
                failures.append(dict(path=name, reason="Size missing or exceeds metadata limit"))
                continue
            url = base + "/" + model + "/resolve/" + revision + "/" + quote(name, safe="/")
            try:
                data = fetch(url, token)
                texts.append(dict(path=name, url=url, sha256=hashlib.sha256(data).hexdigest(),
                                  text=data.decode("utf-8")))
            except Exception as exc:
                failures.append(dict(path=name, reason=str(exc)))
    return dict(model=model, revision=revision, files=files, metadata_texts=texts,
                unread_metadata=failures, card_data=pinned.get("cardData"),
                provenance_status="UNRESOLVED: checkpoint release does not establish local embedding provenance")


def collect_github(repository):
    from urllib.parse import quote
    token = os.environ.get("GITHUB_TOKEN")
    api = "https://api.github.com/repos/" + repository
    info = get_json(api, token)
    branch = info["default_branch"]
    commit = get_json(api + "/commits/" + quote(branch, safe=""), token)["sha"]
    tree = get_json(api + "/git/trees/" + commit + "?recursive=1", token)
    if tree.get("truncated"):
        raise ValueError("GitHub tree is truncated; cannot establish complete inventory")
    entries = [x for x in tree["tree"] if x["type"] == "blob"]
    texts, failures = [], []
    for entry in entries:
        name = entry["path"]
        relevant = (repository.startswith("tesorai/") or name.startswith(("scripts/depmap/", "scripts/msigdb/", "configs/")) or name == "README.md")
        if not relevant or not name.endswith((".py", ".md", ".json", ".yaml", ".yml", ".sh")):
            continue
        if entry.get("size", MAX_TEXT_BYTES + 1) > MAX_TEXT_BYTES:
            failures.append(dict(path=name, reason="Exceeds metadata limit"))
            continue
        url = "https://raw.githubusercontent.com/" + repository + "/" + commit + "/" + quote(name, safe="/")
        try:
            data = fetch(url)
            text = data.decode("utf-8")
            hits = [dict(line=i, text=line) for i, line in enumerate(text.splitlines(), 1) if PATTERN.search(line)]
            texts.append(dict(path=name, url=url, sha256=hashlib.sha256(data).hexdigest(), text=text, matches=hits))
        except Exception as exc:
            failures.append(dict(path=name, reason=str(exc)))
    return dict(repository=repository, revision=commit, files=entries, source_texts=texts,
                unread_sources=failures, provenance_status="Evidence only; no filename-to-checkpoint inference")


def git_state(path):
    if not path.is_dir():
        return dict(path=str(path), status="MISSING")
    result = {}
    for name, args in [("revision", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"]), ("remotes", ["remote", "-v"])]:
        try:
            process = subprocess.run(["git", "-C", str(path)] + args, capture_output=True, text=True, timeout=20)
            result[name] = process.stdout.strip() if process.returncode == 0 else process.stderr.strip()
        except Exception as exc:
            result[name] = str(exc)
    return dict(path=str(path), **result)


def collect_local_sources(roots):
    records, skipped = [], []
    for root in roots:
        if not root.is_dir():
            skipped.append(dict(path=str(root), reason="Directory missing"))
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(part in {".git", "__pycache__", "results", ".venv", "venvs"} for part in path.relative_to(root).parts):
                continue
            is_manifest = path.suffix == ".csv" and re.search(r"manifest|provenance|mapping", path.name, re.I)
            if path.suffix not in {".py", ".md", ".json", ".yaml", ".yml", ".sh"} and not is_manifest:
                continue
            try:
                if path.stat().st_size > MAX_TEXT_BYTES:
                    skipped.append(dict(path=str(path), reason="Text exceeds metadata limit"))
                    continue
                data = path.read_bytes()
                text = data.decode("utf-8-sig")
                records.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), text=text,
                                    matches=[dict(line=i, text=line) for i, line in enumerate(text.splitlines(), 1) if PATTERN.search(line)]))
            except Exception as exc:
                skipped.append(dict(path=str(path), reason=str(exc)))
    return records, skipped


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def inspect_h5ad(path):
    import h5py
    record = dict(path=str(path), size=path.stat().st_size, sha256=sha256_file(path),
                  provenance_status="UNRESOLVED: inspect explicit metadata; names are not proof")
    with h5py.File(path, "r") as f:
        record["root_attributes"] = {k: json_value(v) for k, v in f.attrs.items()}
        record["X_shape"] = list(f["X"].shape) if isinstance(f.get("X"), h5py.Dataset) else None
        record["obsm_keys"] = list(f["obsm"]) if "obsm" in f else []
        metadata, skipped = {}, []
        if "uns" in f:
            def visit(name, node):
                item = dict(attributes={k: json_value(v) for k, v in node.attrs.items()})
                if isinstance(node, h5py.Dataset):
                    # Variable-length strings may be larger than itemsize suggests.
                    if node.size > 10000 or node.size * node.dtype.itemsize > MAX_TEXT_BYTES:
                        skipped.append(dict(path="uns/" + name, reason="Metadata dataset exceeds inspection limit"))
                        return
                    value = json_value(node[()])
                    if len(json.dumps(value)) > MAX_TEXT_BYTES:
                        skipped.append(dict(path="uns/" + name, reason="Decoded metadata exceeds inspection limit"))
                        return
                    item["value"] = value
                metadata[name] = item
            f["uns"].visititems(visit)
            record["uns_attributes"] = {k: json_value(v) for k, v in f["uns"].attrs.items()}
        record["uns_metadata"] = metadata
        record["unread_metadata"] = skipped
    return record


def run(repo=REPO, upstream=UPSTREAM, benchmarks=BENCHMARKS, out=OUT, online=True):
    repo, upstream, benchmarks, out = map(Path, (repo, upstream, benchmarks, out))
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status="UNRESOLVED", local_git=[git_state(repo), git_state(upstream)],
                  remote_hf=[], remote_github=[], errors=[],
                  required_evidence=["Explicit embedding-to-checkpoint manifest", "Immutable checkpoint revision and SHA-256", "Generation source revision, configuration, input hashes and tokenization/pooling settings", "Documented training corpus/modality and base checkpoint"],
                  policy="Collect evidence only. No weights downloaded, pickle loaded, or checkpoint identity inferred from filenames. Published hashes are not locally verified weights.")
    report["local_sources"], report["unread_local_sources"] = collect_local_sources([repo, upstream, benchmarks / "depmap/misc", benchmarks / "msigdb"])
    paths = sorted((benchmarks / "depmap/cell-embs").glob("*.h5ad"))
    msigdb = benchmarks / "msigdb/gene_embs/embs_adata.h5ad.gz"
    if msigdb.is_file():
        paths.append(msigdb)
    else:
        report["errors"].append(dict(path=str(msigdb), reason="MSigDB artifact missing"))
    if not paths:
        report["errors"].append(dict(path=str(benchmarks), reason="No embedding artifacts found"))
    report["embeddings"] = []
    for path in paths:
        try:
            report["embeddings"].append(inspect_h5ad(path))
        except Exception as exc:
            report["errors"].append(dict(path=str(path), reason=str(exc)))
    if online:
        for model in HF_MODELS:
            try:
                report["remote_hf"].append(collect_hf(model))
            except Exception as exc:
                report["errors"].append(dict(source=model, reason=str(exc)))
        for repository in GITHUB_REPOS:
            try:
                report["remote_github"].append(collect_github(repository))
            except Exception as exc:
                report["errors"].append(dict(source=repository, reason=str(exc)))
    else:
        report["errors"].append(dict(source="remote", reason="Online checks disabled"))
    # Match a local artifact hash only to an explicitly published file hash.
    published = {f["published_lfs_sha256"]: (m["model"], m["revision"], f["path"]) for m in report["remote_hf"] for f in m["files"] if f.get("published_lfs_sha256")}
    for artifact in report["embeddings"]:
        match = published.get(artifact["sha256"])
        artifact["exact_published_file_match"] = list(match) if match else None
        # Even an exact file match alone does not prove which checkpoint generated it.
    report["complete_evidence_read"] = not (report["errors"] or report["unread_local_sources"] or any(x["unread_metadata"] for x in report["embeddings"]) or any(x["unread_metadata"] for x in report["remote_hf"]) or any(x["unread_sources"] for x in report["remote_github"]))
    out.mkdir(parents=True, exist_ok=True)
    target = out / "report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for artifact in report["embeddings"]:
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError("Input changed during audit: " + artifact["path"])
    assert json.loads(target.read_text(encoding="utf-8")) == report
    print("Embedding artifacts inspected:", len(report["embeddings"]))
    print("Remote checkpoint repositories inspected:", len(report["remote_hf"]))
    print("Remote source repositories inspected:", len(report["remote_github"]))
    print("Errors or unavailable sources:", len(report["errors"]))
    print("Provenance status: UNRESOLVED; report contains evidence for manual contract verification")
    print(target.resolve())
    return report


# Sequential entry point; functions remain importable for validation.
if globals().get("__name__") == "__main__":
    run()
