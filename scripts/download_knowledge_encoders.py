"""Download and verify the two pinned BindFormer text encoder candidates."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
     "090663c3ae57bf35ffe4d0d468a2a88d03051a4d", "sapbert"),
    ("FremyCompany/BioLORD-2023",
     "167aab527b238a50ca65224e6319215d2ff4fc9f", "biolord-2023"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for repo, revision, name in MODELS:
        destination = ROOT / "assets/models/knowledge" / name
        manifest_path = destination / "asset_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if (existing["repo_id"], existing["revision"]) != (repo, revision):
                raise ValueError(f"Existing asset has another identity: {destination}")
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=False)
        selected = [entry for entry in info.siblings
                    if entry.rfilename.endswith((".json", ".txt"))
                    or entry.rfilename in ("README.md", "model.safetensors")]
        if not any(entry.rfilename == "model.safetensors" for entry in selected):
            raise ValueError(f"No safetensors weights in {repo}")
        print(f"Downloading {repo} at {revision} -> {destination}", flush=True)
        snapshot_download(repo, revision=revision, token=False, local_dir=destination,
                          allow_patterns=[entry.rfilename for entry in selected], max_workers=3)
        files = []
        for entry in selected:
            path = destination / entry.rfilename
            if entry.size is not None and path.stat().st_size != entry.size:
                raise ValueError(f"Size mismatch: {path}")
            checksum = sha256(path)
            if entry.lfs and checksum != entry.lfs.sha256:
                raise ValueError(f"Hub LFS SHA256 mismatch: {path}")
            files.append({"path": entry.rfilename, "size": path.stat().st_size,
                          "sha256": checksum})
        manifest = {"repo_id": repo, "revision": revision, "gated": info.gated,
                    "files": sorted(files, key=lambda item: item["path"])}
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        print(f"Verified {repo}: {len(files)} files, "
              f"{sum(item['size'] for item in files):,} bytes", flush=True)


if __name__ == "__main__":
    main()
