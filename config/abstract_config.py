from abc import ABC

__all__ = ['AbstractConfig']


class AbstractConfig(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.PREFETCH_FACTOR = 2
