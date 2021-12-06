from .abstract_config import AbstractConfig

__all__ = ['TrainConfig']


class TrainConfig(AbstractConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_config()
        self.data_config()

    def data_config(self):
        self.BATCH_SIZE = 4
        self.NUM_WORKERS = 6

    def model_config(self):
        self.IMAGE_SIZE = (1100, 1100)
