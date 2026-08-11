from __future__ import annotations

from typing import Any


class PipelineError(RuntimeError):
    """A stable machine-readable pipeline failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "details": self.details}
