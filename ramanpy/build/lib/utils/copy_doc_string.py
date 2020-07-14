def copy_doc_string(func_copy_from):
    """

    Decorator to copy docstrings

    
    """
    def inner(func_copy_to):
        func_copy_to.__doc__ = func_copy_from.__doc__
        return func_copy_to
    return inner

