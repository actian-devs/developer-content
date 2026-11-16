"""Exception types. Each carries the process exit code the CLI should use."""

from __future__ import annotations

EXIT_OK = 0
EXIT_CHECK_FAILED = 1  # a check failed / blocking finding / runtime failure
EXIT_USAGE = 2         # bad arguments or unmet precondition
EXIT_INTERRUPTED = 3   # stopped by SIGINT/SIGTERM; state saved, safe to --resume


class MigrationError(Exception):
    exit_code = EXIT_CHECK_FAILED


class UsageError(MigrationError):
    """Bad arguments or an unmet precondition (nothing was changed)."""

    exit_code = EXIT_USAGE


class SourceError(MigrationError):
    """Reading from Chroma failed after retries."""


class TargetError(MigrationError):
    """Writing to / reading from Actian failed after retries."""


class CapacityError(TargetError):
    """Actian refused writes because the licence capacity is exhausted."""


class Interrupted(MigrationError):
    exit_code = EXIT_INTERRUPTED
