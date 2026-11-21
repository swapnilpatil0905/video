"""Platform error hierarchy. Each error maps to a documented CLI exit code."""

from __future__ import annotations


class PlatformError(Exception):
    """Base class for all expected platform errors."""

    exit_code = 1


class ConfigError(PlatformError):
    """Invalid or incomplete configuration (exit code 2)."""

    exit_code = 2


class PolicyViolation(ConfigError):
    """A configured value violates a platform policy (e.g. a forbidden host)."""


class IllegalStateError(PlatformError):
    """The requested command is not valid in the project's current state (exit code 3)."""

    exit_code = 3


class ServiceUnavailableError(PlatformError):
    """A required external service could not be reached (exit code 4)."""

    exit_code = 4


class StageFailure(PlatformError):
    """A pipeline stage failed (exit code 5)."""

    exit_code = 5


class NotImplementedYet(PlatformError):
    """The feature is defined in the CLI contract but belongs to a later milestone (exit code 6)."""

    exit_code = 6

    def __init__(self, feature: str, milestone: str) -> None:
        super().__init__(f"'{feature}' is not implemented yet (planned for milestone {milestone}).")
        self.feature = feature
        self.milestone = milestone
