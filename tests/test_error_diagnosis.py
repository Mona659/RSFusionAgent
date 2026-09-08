from rsfusion_agent.tools.error_diagnosis import diagnose_error


def test_native_runtime_diagnosis_is_retryable() -> None:
    diagnosis = diagnose_error(RuntimeError("Model runtime failed with exit code 3221226505"))

    assert diagnosis.category == "native_runtime"
    assert diagnosis.retryable is True
    assert "Retry once" in diagnosis.recommended_action


def test_missing_path_diagnosis_does_not_expose_the_path() -> None:
    diagnosis = diagnose_error(FileNotFoundError("Checkpoint does not exist: D:/private/model.pth"))

    assert diagnosis.category == "configuration"
    assert diagnosis.retryable is False
    assert "D:/private" not in diagnosis.summary
