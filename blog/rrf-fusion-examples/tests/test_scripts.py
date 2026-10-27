"""Each script must print exactly what the article shows."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"

EXPECTED = {
    "01_score_scale_example.py": """\
pooling-guide: 0.03252
error-guide: 0.03227
timeout-faq: 0.03200
""",
    "02_k_sensitivity.py": """\
k=60  exact: 0.01639  general: 0.03125  higher: General guide
k=10  exact: 0.09091  general: 0.14286  higher: General guide
k=2   exact: 0.33333  general: 0.33333  higher: Tie
k=1   exact: 0.50000  general: 0.40000  higher: Exact error guide
""",
    "03_parent_deduplication.py": """\
Chunks
  pool-1: 0.03252
  pool-2: 0.03252
  error-1: 0.03150
  faq-1: 0.03150
Parents
  pool-guide: 0.03279
  error-guide: 0.03200
  timeout-faq: 0.03200
""",
    "05_fusion_ranking_difference.py": """\
RRF
pooling-guide: 0.03252
timeout-faq: 0.03200
exact-error-guide: 0.03154
retry-policy: 0.03150
connection-limits: 0.03101
network-diagnostics: 0.03054
DBSF
pooling-guide: 0.59788
exact-error-guide: 0.54336
timeout-faq: 0.54141
retry-policy: 0.48849
connection-limits: 0.43912
network-diagnostics: 0.38975
""",
}


@pytest.mark.parametrize("script", EXPECTED)
def test_script_output_matches_article(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script)], capture_output=True, text=True, check=True
    )
    assert result.stdout.replace("\r\n", "\n") == EXPECTED[script]
