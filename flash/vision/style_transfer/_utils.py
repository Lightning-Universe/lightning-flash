from typing import NoReturn

__all__ = ["raise_not_supported"]


def raise_not_supported(phase: str) -> NoReturn:
    raise RuntimeError(f"Style transfer does not support a {phase} phase.")
