def is_jupyter_notebook():
    """
    Checks whether the current environment is a Jupyter Notebook.

    Returns:
        bool: True if the code is running in a Jupyter Notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        if "IPKernelApp" in get_ipython().config:
            return True
        else:
            return False
    except ImportError:
        return False


def get_tqdm():
    """
    Returns the appropriate tqdm function based on the environment.

    Returns:
        tqdm: The appropriate tqdm function for the current environment (Jupyter Notebook or standard Python script).
    """
    if is_jupyter_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm


def main():
    print(get_tqdm())


if __name__ == "__main__":
    main()
