"""Benchmark loaders: two source tables plus the gold set of true matches.

Datasets come from the Leipzig DB group's entity resolution benchmark collection
(https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).
Both pair a left and a right table; every true match links one left record to one right record.
"""
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")


@dataclass(frozen=True)
class Spec:
    url: str
    left_file: str
    right_file: str
    gold_file: str
    left_name: str
    right_name: str
    left_fields: tuple[str, ...]     # fields flattened into the text that gets embedded
    right_fields: tuple[str, ...]


DATASETS = {
    "abt-buy": Spec(
        url="https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip",
        left_file="Abt.csv", right_file="Buy.csv", gold_file="abt_buy_perfectMapping.csv",
        left_name="abt", right_name="buy",
        left_fields=("name", "description", "price"),
        right_fields=("name", "description", "manufacturer", "price"),
    ),
    "amazon-google": Spec(
        url="https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip",
        left_file="Amazon.csv", right_file="GoogleProducts.csv",
        gold_file="Amzon_GoogleProducts_perfectMapping.csv",   # sic: the file name in the archive
        left_name="amazon", right_name="google",
        left_fields=("title", "description", "manufacturer", "price"),
        right_fields=("name", "description", "manufacturer", "price"),
    ),
}


@dataclass
class Dataset:
    name: str
    spec: Spec
    left: pd.DataFrame       # columns: id, text, plus the raw fields
    right: pd.DataFrame
    gold: set[tuple[str, str]]   # (left_id, right_id)


def _download(spec: Spec, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):   # the Leipzig server is slow and sometimes times out
        try:
            r = requests.get(spec.url, timeout=(30, 600))
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise SystemExit(f"Download failed ({e}).\nDownload {spec.url} manually and put "
                                 f"{spec.left_file}, {spec.right_file} and {spec.gold_file} in {target}/") from e
            print(f"  download attempt {attempt + 1} failed, retrying...")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for member in z.namelist():
            name = Path(member).name
            if name in (spec.left_file, spec.right_file, spec.gold_file):
                (target / name).write_bytes(z.read(member))


def _read(path: Path) -> pd.DataFrame:
    # the Leipzig CSVs are Latin-1, not UTF-8
    df = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def to_text(row: pd.Series, fields: tuple[str, ...]) -> str:
    """'name: Sony Bravia 40in LCD TV | price: $899.99' (empty fields are skipped)."""
    return " | ".join(f"{f}: {row[f].strip()}" for f in fields if f in row and row[f].strip())


def load(name: str) -> Dataset:
    spec = DATASETS[name]
    folder = DATA_DIR / name
    if not all((folder / f).exists() for f in (spec.left_file, spec.right_file, spec.gold_file)):
        _download(spec, folder)

    left, right = _read(folder / spec.left_file), _read(folder / spec.right_file)
    for df, fields in ((left, spec.left_fields), (right, spec.right_fields)):
        df["id"] = df["id"].str.strip()
        df["text"] = df.apply(to_text, axis=1, fields=fields)

    gold_df = _read(folder / spec.gold_file)
    gold = {(a.strip(), b.strip()) for a, b in zip(gold_df.iloc[:, 0], gold_df.iloc[:, 1])}
    return Dataset(name, spec, left, right, gold)
