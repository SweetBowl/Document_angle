from abc import ABC, abstractclassmethod


class AbstractMeter(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    @abstractclassmethod
    def reset(self):
        msg = 'reset should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def update(self):
        msg = 'update should be implemented by subclass.'
        raise NotImplementedError(msg)

    def update_value(self):
        if self._value.keys() != self.sum.keys():
            msg = 'self._value and self.sum should have same keys.'
            raise ValueError(msg)

        keys = self._value.keys()

        for key in keys:
            self._value[key] = self.sum[key] / self.count

    @property
    def value(self) -> dict:
        return self._value

    def __str__(self):
        items = [
            ': '.join([name, '{:.6}'.format(value)])
            for (name, value) in self._value.items()
        ]
        return ', '.join(items)
