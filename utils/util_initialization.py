__all__ = ['initialize_meters', 'initialize_optimizers']


def initialize_meters(meters):
    if 'train' in meters.keys():
        for meter_name in meters['train'].keys():
            meters['train'][meter_name].reset()

    if 'val' in meters.keys():
        for meter_name in meters['val'].keys():
            meters['val'][meter_name].reset()

    if 'test' in meters.keys():
        for meter_name in meters['test'].keys():
            meters['test'][meter_name].reset()


def initialize_optimizers(optimizers):
    for optimizer_name in optimizers.keys():
        optimizers[optimizer_name].zero_grad()
