__all__ = ['variable']


def variable(func):
    '''
    A decorator. Make a function look like a variable.

    Args:
        func(function)
    '''
    return func()
