from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

IMAGE = {".png", ".jpg", ".jpeg"}
FOLDERS = {"pages": ("manual_page", IMAGE), "photos": ("photo", IMAGE),
           "video": ("video", {".mp4"}), "voice": ("voice", {".wav", ".flac"})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="public_data")
    ap.add_argument("--dpi", type=int, default=144, help="resolution for rendered PDF pages")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}. Create it and add assets under "
            f"{data_dir}/<asset_id>/ before generating the manifest."
        )
    asset_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not asset_dirs:
        raise SystemExit(
            f"No asset directories found in {data_dir}. Add assets under "
            f"{data_dir}/<asset_id>/ before generating the manifest."
        )

    manifest, pages = [], []
    for asset_dir in asset_dirs:
        asset_id = asset_dir.name

        # PDFs: text per page, plus each page rendered to pages/<pdf>_pNNN.png
        pdfs = sorted((asset_dir / "manual").glob("*.pdf"))
        for pdf in pdfs:
            source_metadata = asset_dir / "source.json"
            source_url = (
                json.loads(source_metadata.read_text()).get("source_url", pdf.name)
                if source_metadata.exists()
                else pdf.name
            )
            (asset_dir / "pages").mkdir(exist_ok=True)
            with pymupdf.open(pdf) as doc:
                for n, page in enumerate(doc, start=1):
                    text = page.get_text().strip()
                    if text:
                        manifest.append({"asset_id": asset_id, "modality": "manual_text",
                                         "text": text, "source": f"{source_url}#page={n}"})
                    page.get_pixmap(dpi=args.dpi).save(asset_dir / "pages" / f"{pdf.stem}_p{n:03d}.png")

        manual = asset_dir / "manual.txt"
        if manual.exists() and not pdfs:
            source_metadata = asset_dir / "source.json"
            source = (
                json.loads(source_metadata.read_text()).get("source_url", "manual.txt")
                if source_metadata.exists()
                else "manual.txt"
            )
            manifest.append({"asset_id": asset_id, "modality": "manual_text",
                             "text": manual.read_text(), "source": source})

        for folder, (modality, suffixes) in FOLDERS.items():
            for f in sorted((asset_dir / folder).glob("*")):
                if f.suffix.lower() in suffixes:
                    manifest.append({"asset_id": asset_id, "modality": modality, "path": str(f)})
                    if modality == "manual_page":
                        pages.append({"asset_id": asset_id, "path": str(f)})

    if not manifest:
        raise SystemExit(
            f"No supported files found under {data_dir}. Add a manual PDF or TXT, "
            "images under pages/ or photos/, MP4 files under video/, or WAV/FLAC "
            "files under voice/."
        )

    Path("manifest.json").write_text(json.dumps(manifest, indent=2))
    Path("pages.json").write_text(json.dumps(pages, indent=2))
    counts: dict[str, int] = {}
    for rec in manifest:
        counts[rec["modality"]] = counts.get(rec["modality"], 0) + 1
    print(f"assets: {len({r['asset_id'] for r in manifest})}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<12} {v}")
    print(f"wrote manifest.json ({len(manifest)} records) and pages.json ({len(pages)} pages)")


if __name__ == "__main__":
    main()