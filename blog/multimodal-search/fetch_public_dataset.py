from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from urllib.error import URLError
from urllib.request import Request, urlopen

REVISION = "7376533ab86ea07acc77ef6a9f3c3e57fcff833d"
REPOSITORY = "https://github.com/raspberrypi/documentation"
RAW_ROOT = f"https://raw.githubusercontent.com/raspberrypi/documentation/{REVISION}"
DESTINATION = Path("public_data")

DOCUMENTS = (
    ("pi-ai-hat-plus", "Raspberry Pi AI HAT+", "documentation/asciidoc/accessories/ai-hat-plus/about.adoc"),
    ("pi-audio-configuration", "Raspberry Pi audio board configuration", "documentation/asciidoc/accessories/audio/configuration.adoc"),
    ("pi-audio-getting-started", "Raspberry Pi audio board getting started", "documentation/asciidoc/accessories/audio/getting_started.adoc"),
    ("pi-audio-hardware-info", "Raspberry Pi audio board hardware information", "documentation/asciidoc/accessories/audio/hardware-info.adoc"),
    ("pi-audio-overview", "Raspberry Pi audio boards overview", "documentation/asciidoc/accessories/audio/introduction.adoc"),
    ("pi-audio-firmware", "Raspberry Pi audio board firmware update", "documentation/asciidoc/accessories/audio/update-firmware.adoc"),
    ("pi-build-hat-motors", "Raspberry Pi Build HAT motors for .NET", "documentation/asciidoc/accessories/build-hat/net-motors.adoc"),
    ("pi-build-hat-sensors", "Raspberry Pi Build HAT sensors for .NET", "documentation/asciidoc/accessories/build-hat/net-sensors.adoc"),
    ("pi-codec-zero", "Raspberry Pi Codec Zero", "documentation/asciidoc/accessories/audio/codec_zero.adoc"),
    ("pi-dac-pro", "Raspberry Pi DAC Pro", "documentation/asciidoc/accessories/audio/dac_pro.adoc"),
    ("pi-digiamp-plus", "Raspberry Pi DigiAMP+", "documentation/asciidoc/accessories/audio/digiamp_plus.adoc"),
    ("pi-microcontrollers", "Raspberry Pi microcontroller chips", "documentation/asciidoc/microcontrollers/microcontroller-chips.adoc"),
    ("pi-computer-setup", "Set up a Raspberry Pi computer", "documentation/asciidoc/computers/getting-started/setting-up.adoc"),
    ("pi-install-os", "Install an operating system on Raspberry Pi", "documentation/asciidoc/computers/getting-started/install.adoc"),
)

IMAGES = {
    "pi-ai-hat-plus": "images/ai-hat-plus-hero.jpg",
    "pi-audio-configuration": "images/gui.png",
    "pi-audio-getting-started": "images/Chatter_Box.jpg",
    "pi-audio-hardware-info": "images/all_audio_boards_gpio_pinouts.png",
    "pi-audio-firmware": "images/write_protect_tabs.jpg",
    "pi-codec-zero": "images/Codec_Zero_Board_Diagram.jpg",
    "pi-computer-setup": "images/peripherals/microSD.png",
    "pi-dac-pro": "images/DAC_Pro_Board_Diagram.jpg",
    "pi-digiamp-plus": "images/DigiAMP+_Board_Diagram.jpg",
    "pi-install-os": "images/imager/device-tab.png",
}


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "multimodal-search-public-dataset/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except URLError as exc:
        raise SystemExit(f"Could not download {url}: {exc}") from exc


def main() -> None:
    documents = []
    for asset_id, title, source_path in DOCUMENTS:
        source_url = f"{REPOSITORY}/blob/{REVISION}/{source_path}"
        body = fetch(f"{RAW_ROOT}/{quote(source_path, safe='/')}").decode("utf-8").strip()
        if not body:
            raise SystemExit(f"Downloaded an empty source document: {source_url}")
        image_path = IMAGES.get(asset_id)
        image_source = (
            f"{RAW_ROOT}/{quote(str(Path(source_path).parent / image_path), safe='/')}"
            if image_path
            else None
        )
        image = fetch(image_source) if image_source else None
        documents.append((asset_id, title, source_path, source_url, body, image_path, image_source, image))
    license_text = fetch(f"{RAW_ROOT}/LICENSE.md").decode("utf-8")

    expected_entries = {asset_id for asset_id, _, _ in DOCUMENTS}
    expected_entries.update({"DATASET.md", "UPSTREAM_LICENSE.md"})
    if DESTINATION.exists():
        unexpected = {path.name for path in DESTINATION.iterdir()} - expected_entries
        if unexpected:
            raise SystemExit(
                f"Refusing to overwrite unexpected files in {DESTINATION}: "
                f"{', '.join(sorted(unexpected))}"
            )
        for asset_id, _, _ in DOCUMENTS:
            asset_dir = DESTINATION / asset_id
            if asset_dir.exists():
                allowed = {
                    "manual.txt", "source.json", "manual", "pages", "photos", "video",
                    "voice", "queries",
                }
                unexpected = {path.name for path in asset_dir.iterdir()} - allowed
                if unexpected:
                    raise SystemExit(
                        f"Refusing to overwrite unexpected files in {asset_dir}: "
                        f"{', '.join(sorted(unexpected))}"
                    )
            image_path = IMAGES.get(asset_id)
            if image_path:
                photo_file = DESTINATION / asset_id / "photos" / Path(image_path).name
                photo_dir = photo_file.parent
                if photo_dir.exists():
                    unexpected = {path.name for path in photo_dir.iterdir()} - {photo_file.name}
                    if unexpected:
                        raise SystemExit(
                            f"Refusing to overwrite unexpected files in {photo_dir}: "
                            f"{', '.join(sorted(unexpected))}"
                        )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for asset_id, title, source_path, source_url, body, image_path, image_source, image in documents:
        asset_dir = DESTINATION / asset_id
        asset_dir.mkdir(exist_ok=True)
        (asset_dir / "manual.txt").write_text(f"{title}\n\n{body}\n", encoding="utf-8")
        source_metadata = {
            "title": title,
            "repository": REPOSITORY,
            "revision": REVISION,
            "source_path": source_path,
            "source_url": source_url,
            "license": "CC BY-SA 4.0",
        }
        if image_path and image_source and image is not None:
            image_filename = Path(image_path).name
            photo_dir = asset_dir / "photos"
            photo_dir.mkdir(exist_ok=True)
            (photo_dir / image_filename).write_bytes(image)
            source_metadata["image_path"] = image_path
            source_metadata["image_url"] = image_source
            source_metadata["image_local_path"] = str(photo_dir / image_filename)
        (asset_dir / "source.json").write_text(
            json.dumps(source_metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    (DESTINATION / "UPSTREAM_LICENSE.md").write_text(license_text, encoding="utf-8")
    rows = []
    for (
        asset_id, title, source_path, source_url, _body, _image_path, image_url, _image
    ) in documents:
        image_link = f"[image]({image_url})" if image_url else "—"
        rows.append(
            f"| `{asset_id}` | {title} | [{source_path}]({source_url}) | {image_link} |"
        )
    (DESTINATION / "DATASET.md").write_text(
        "# Public Raspberry Pi documentation sample\n\n"
        f"Source repository: [{REPOSITORY}]({REPOSITORY}) at commit "
        f"[`{REVISION}`]({REPOSITORY}/commit/{REVISION}).\n\n"
        "The source repository identifies its documentation as CC BY-SA 4.0. "
        "The upstream license text is included in [UPSTREAM_LICENSE.md](UPSTREAM_LICENSE.md). "
        "The `manual.txt` files reproduce the listed AsciiDoc source with a title heading "
        "added. Selected image files are copied from each document's `images/` folder; "
        "their pinned source URLs are recorded in `source.json`. Retain this attribution "
        "and license when redistributing these materials.\n\n"
        "| Asset ID | Title | Source document | Image source |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n"
        "The generated PDF layout, video clips, and eSpeak NG narration are created by "
        "`prepare_public_media.py`; they are derived from these CC BY-SA documents and "
        "images for benchmark testing. The voice question recordings are synthesized "
        "from the original query text. The generated videos are short still-page clips, "
        "not source footage.\n",
        encoding="utf-8",
    )
    image_count = sum(document[-1] is not None for document in documents)
    print(
        f"Downloaded {len(documents)} attributed Raspberry Pi documents and "
        f"{image_count} official images to {DESTINATION}/"
    )


if __name__ == "__main__":
    main()
