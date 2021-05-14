from typing import NoReturn

__all__ = ["raise_not_supported"]


def raise_not_supported(phase: str) -> NoReturn:
    raise RuntimeError(
        f"Style transfer does not support a {phase} phase, "
        f"since there is no metric to objectively determine the quality of a stylization."
    )
