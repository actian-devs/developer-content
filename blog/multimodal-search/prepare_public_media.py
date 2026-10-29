from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import fitz
from PIL import Image

DATA_DIR = Path("public_data")
QUERIES_PATH = Path("queries.json")
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
PAGE_SIZE = fitz.paper_rect("a4")


def require_command(name: str) -> str:
    command = shutil.which(name)
    if command is None:
        raise SystemExit(
            f"Required command '{name}' is not installed. Install ffmpeg and espeak-ng "
            "before preparing the public multimodal sample."
        )
    return command


def plain_text(text: str) -> str:
    text = re.sub(r"(?m)^include::.*$", "", text)
    text = re.sub(r"(?m)^image::.*$", "", text)
    text = re.sub(r"(?m)^video::.*$", "", text)
    text = re.sub(r"xref:[^\[]+\[([^\]]*)\]", r"\1", text)
    text = re.sub(r"https?://\S+\[([^\]]*)\]", r"\1", text)
    text = re.sub(r"(?m)^\[\[.*\]\]\s*$", "", text)
    text = re.sub(r"(?m)^[-=]{3,}\s*$", "", text)
    text = re.sub(r"(?m)^\[.*\]\s*$", "", text)
    text = re.sub(r"(?m)^[=]+\s*", "", text)
    text = re.sub(r"(?m)^\|=+$", "", text)
    text = re.sub(r"(?m)^\|", "", text)
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\{plus\}", "+", text)
    text = re.sub(r" {2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def write_pdf(asset_dir: Path, title: str, source_url: str, body: str) -> None:
    manual_dir = asset_dir / "manual"
    manual_dir.mkdir(exist_ok=True)
    output = manual_dir / f"{asset_dir.name}.pdf"
    source = json.loads((asset_dir / "source.json").read_text())
    image_path = Path(source["image_local_path"]) if source.get("image_local_path") else None
    content = plain_text(body)
    attribution = f"Source: {source_url} (CC BY-SA 4.0; adapted for this benchmark)"
    if source.get("image_url"):
        attribution += f"\nImage source: {source['image_url']} (CC BY-SA 4.0)"

    doc = fitz.open()
    page = doc.new_page(width=PAGE_SIZE.width, height=PAGE_SIZE.height)
    page.insert_font(fontname="DejaVu", fontfile=str(FONT_PATH))
    page.insert_text((48, 48), title, fontname="DejaVu", fontsize=18)
    attribution_space = page.insert_textbox(
        fitz.Rect(48, 58, PAGE_SIZE.width - 48, 108),
        attribution,
        fontname="DejaVu",
        fontsize=5.5,
        color=(0.25, 0.25, 0.25),
    )
    if attribution_space < 0:
        raise ValueError(f"Attribution does not fit on the PDF title page for {asset_dir.name}")
    y = 112
    if image_path and image_path.is_file():
        with Image.open(image_path) as image:
            width, height = image.size
        scale = min((PAGE_SIZE.width - 96) / width, 210 / height)
        rect = fitz.Rect(
            48,
            y,
            48 + width * scale,
            y + height * scale,
        )
        page.insert_image(rect, filename=str(image_path), keep_proportion=True)
        y = rect.y1 + 20
    y = write_text_pages(doc, page, content, y)
    if y == 112:
        doc.new_page(width=PAGE_SIZE.width, height=PAGE_SIZE.height)
    doc.set_metadata(
        {
            "title": title,
            "author": "Raspberry Pi Ltd.",
            "subject": f"Adapted from {source_url}; CC BY-SA 4.0",
            "keywords": "Raspberry Pi, public benchmark, CC BY-SA 4.0",
        }
    )
    doc.save(output, garbage=4, deflate=True)
    doc.close()


def write_text_pages(doc: fitz.Document, page: fitz.Page, content: str, y: float) -> float:
    line_height = 13
    x = 48
    max_width = PAGE_SIZE.width - 96
    font = fitz.Font(fontfile=str(FONT_PATH))
    for paragraph in content.splitlines():
        words = paragraph.split()
        line = ""
        wrapped = []
        for word in words:
            candidate = f"{line} {word}".strip()
            if line and font.text_length(candidate, fontsize=9) > max_width:
                wrapped.append(line)
                line = word
            else:
                line = candidate
        if line:
            wrapped.append(line)
        for line in wrapped:
            if y > PAGE_SIZE.height - 54:
                page = doc.new_page(width=PAGE_SIZE.width, height=PAGE_SIZE.height)
                page.insert_font(fontname="DejaVu", fontfile=str(FONT_PATH))
                y = 48
            page.insert_text((x, y), line, fontname="DejaVu", fontsize=9)
            y += line_height
        y += 6
    return y


def run_espeak(executable: str, ffmpeg: str, text: str, output: Path) -> None:
    source = output.with_name(f"{output.stem}-source.wav")
    try:
        subprocess.run(
            [executable, "-s", "145", "-w", str(source), "--", text],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        source.unlink(missing_ok=True)


def write_video(ffmpeg: str, page_image: Path, output: Path) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "1",
            "-i",
            str(page_image),
            "-t",
            "3",
            "-vf",
            "scale=320:320:force_original_aspect_ratio=decrease,"
            "pad=320:320:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def first_text_query(queries: list[dict], asset_id: str) -> str:
    for query in queries:
        if (
            query["query_modality"] == "text"
            and query["target_modality"] == "manual_text"
            and asset_id in query["relevant"]
        ):
            return query["query"]
    raise ValueError(f"No text-to-manual_text question is labelled for {asset_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build attributed PDFs, generated audio, and page-based video clips."
    )
    parser.add_argument(
        "--skip-voice",
        action="store_true",
        help="build PDFs and videos without speech files or voice-query records",
    )
    args = parser.parse_args()

    espeak = None if args.skip_voice else require_command("espeak-ng")
    ffmpeg = require_command("ffmpeg")
    if not FONT_PATH.is_file():
        raise SystemExit(f"Required font not found: {FONT_PATH}")
    if not DATA_DIR.is_dir():
        raise SystemExit("Run `uv run fetch_public_dataset.py` before this script.")

    queries = json.loads(QUERIES_PATH.read_text())
    base_queries = [
        query for query in queries
        if query["query_modality"] == "text"
        and query["target_modality"] == "manual_text"
    ]
    generated_queries = []

    for asset_dir in sorted(path for path in DATA_DIR.iterdir() if path.is_dir()):
        metadata_path = asset_dir / "source.json"
        manual_path = asset_dir / "manual.txt"
        if not metadata_path.is_file() or not manual_path.is_file():
            continue

        source = json.loads(metadata_path.read_text())
        title = source["title"]
        body = manual_path.read_text(encoding="utf-8")
        asset_id = asset_dir.name
        write_pdf(asset_dir, title, source["source_url"], body)

        readable = plain_text(body)
        if espeak:
            voice_dir = asset_dir / "voice"
            voice_dir.mkdir(exist_ok=True)
            narration = f"{title}. {readable[:900]}"
            voice_path = voice_dir / f"{asset_id}-narration.wav"
            run_espeak(espeak, ffmpeg, narration, voice_path)

        pdf_path = asset_dir / "manual" / f"{asset_id}.pdf"
        with fitz.open(pdf_path) as pdf:
            page_image = asset_dir / "manual" / f"{asset_id}-video-frame.png"
            pdf[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).save(page_image)
        video_dir = asset_dir / "video"
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir / f"{asset_id}-page.mp4"
        write_video(ffmpeg, page_image, video_path)
        page_image.unlink()

        question = first_text_query(base_queries, asset_id)
        relevant = [asset_id]
        generated_queries.extend(
            [
                {
                    "query_modality": "text",
                    "query": question,
                    "target_modality": "manual_page",
                    "relevant": relevant,
                },
                {
                    "query_modality": "text",
                    "query": question,
                    "target_modality": "video",
                    "relevant": relevant,
                },
                {
                    "query_modality": "video",
                    "query": video_path.as_posix(),
                    "target_modality": "manual_page",
                    "relevant": relevant,
                },
            ]
        )
        if espeak:
            voice_query_path = asset_dir / "queries" / f"{asset_id}-question.wav"
            voice_query_path.parent.mkdir(parents=True, exist_ok=True)
            run_espeak(espeak, ffmpeg, question, voice_query_path)
            generated_queries.extend(
                [
                    {
                        "query_modality": "voice",
                        "query": voice_query_path.as_posix(),
                        "target_modality": "manual_text",
                        "relevant": relevant,
                    },
                    {
                        "query_modality": "voice",
                        "query": voice_query_path.as_posix(),
                        "target_modality": "voice",
                        "relevant": relevant,
                    },
                ]
            )

        image_path = source.get("image_local_path")
        if image_path and Path(image_path).is_file():
            generated_queries.append(
                {
                    "query_modality": "photo",
                    "query": image_path,
                    "target_modality": "manual_page",
                    "relevant": relevant,
                }
            )

    if not generated_queries:
        raise SystemExit("No public multimodal assets were prepared.")
    all_queries = base_queries + generated_queries
    missing_queries = [
        query["query"]
        for query in all_queries
        if query["query_modality"] != "text" and not Path(query["query"]).is_file()
    ]
    if missing_queries:
        raise FileNotFoundError(f"Generated query media is missing: {missing_queries[0]}")
    QUERIES_PATH.write_text(
        json.dumps(all_queries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    modalities = sorted({query["query_modality"] for query in generated_queries})
    voice_status = "voice included" if espeak else "voice skipped (--skip-voice)"
    print(
        f"Prepared PDFs and videos; added {len(generated_queries)} evaluation records "
        f"({', '.join(modalities)} queries; {voice_status}). "
        "Run `uv run make_manifest.py` next."
    )


if __name__ == "__main__":
    main()
