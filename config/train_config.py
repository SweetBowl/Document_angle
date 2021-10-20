from .abstract_config import AbstractConfig

__all__ = ['TrainConfig']


class TrainConfig(AbstractConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_config()
        self.data_config()

    def data_config(self):
        self.BATCH_SIZE = 64
        self.NUM_WORKERS = 8

    def model_config(self):
        self.IMAGE_SIZE = (1600, 1600)
