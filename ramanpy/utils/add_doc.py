def add_doc(text):
    """

    Adds docstring

    
    """
    def inner(func):
        func.__doc__ = text
        return func
    return inner