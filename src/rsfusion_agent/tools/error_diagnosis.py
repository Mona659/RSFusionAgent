"""Classify local agent failures into safe, user-actionable diagnostics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ErrorDiagnosis(BaseModel):
    category: Literal[
        "configuration",
        "data_validation",
        "cuda_environment",
        "native_runtime",
        "timeout",
        "llm_configuration",
        "unknown",
    ]
    retryable: bool
    summary: str
    recommended_action: str


def diagnose_error(error: Exception) -> ErrorDiagnosis:
    """Return a redaction-safe diagnosis without exposing configured paths."""

    message = str(error).lower()
    if isinstance(error, FileNotFoundError):
        return ErrorDiagnosis(
            category="configuration",
            retryable=False,
            summary="A configured local file or executable is missing.",
            recommended_action="Check the H5, checkpoint, output and model-Python paths in the local configuration.",
        )
    if "exit code 3221226505" in message or "0xc0000409" in message:
        return ErrorDiagnosis(
            category="native_runtime",
            retryable=True,
            summary="The isolated model process ended with a Windows native fast-fail error.",
            recommended_action="Retry once; if it recurs, inspect CUDA/DLL logs and GPU stability.",
        )
    if "exceeded" in message and ("model inference" in message or "model process" in message):
        return ErrorDiagnosis(
            category="timeout",
            retryable=False,
            summary="The local model process exceeded its configured timeout.",
            recommended_action="Check GPU utilization and increase the timeout only after confirming progress.",
        )
    if "cuda was requested but is not available" in message or "cuda" in message and "not available" in message:
        return ErrorDiagnosis(
            category="cuda_environment",
            retryable=False,
            summary="The configured model environment cannot provide the requested CUDA device.",
            recommended_action="Run preflight-runtime, verify the Conda PyTorch/CUDA environment, or select CPU.",
        )
    if "does not exist" in message or "no such file" in message:
        return ErrorDiagnosis(
            category="configuration",
            retryable=False,
            summary="A configured local file or executable is missing.",
            recommended_action="Check the H5, checkpoint, output and model-Python paths in the local configuration.",
        )
    if any(token in message for token in ("must have shape", "channel", "patch index", "safety gate")):
        return ErrorDiagnosis(
            category="data_validation",
            retryable=False,
            summary="The configured request does not satisfy the YRE-151 data safety checks.",
            recommended_action="Inspect the H5 pair and use the configured patch index and 151-band profile.",
        )
    if "api key" in message or "provider" in message and "configured" in message:
        return ErrorDiagnosis(
            category="llm_configuration",
            retryable=False,
            summary="The LLM provider configuration is incomplete or invalid.",
            recommended_action="Check RSFUSION_LLM_* environment variables without putting keys in source code.",
        )
    return ErrorDiagnosis(
        category="unknown",
        retryable=False,
        summary="The operation failed without a known recovery category.",
        recommended_action="Read the sanitized error and run preflight-runtime before retrying.",
    )
