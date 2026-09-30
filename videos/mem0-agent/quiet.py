"""Silence the import-time noise from mem0's dependencies. Import before mem0."""

import logging
import os
import sys
import warnings

# mem0 ships PostHog telemetry on by default and reads this flag once, at import
os.environ.setdefault("MEM0_TELEMETRY", "False")

# langchain-community announces its own sunsetting the first time it is imported.
# mem0 imports it lazily, deep inside from_config, by which point something has
# reset the warning filters - so import it here, under our own filter, and let
# the later import find it already in sys.modules.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import langchain_community  # noqa: F401
    except ImportError:
        pass

# PostHog and mem0 both log warnings we can't act on (telemetry quotas, no spaCy,
# no keyword search on this store). PostHog resets its own logger level in its
# constructor, so setting a level per logger doesn't stick - this does.
logging.disable(logging.WARNING)

# PostHog's background event loop gets collected half-built at interpreter exit,
# which prints an "Exception ignored in BaseEventLoop.__del__" traceback
_default_unraisable = sys.unraisablehook


def _quiet_unraisable(unraisable):
    if isinstance(unraisable.exc_value, AttributeError) and "_closed" in str(unraisable.exc_value):
        return
    _default_unraisable(unraisable)


sys.unraisablehook = _quiet_unraisable
