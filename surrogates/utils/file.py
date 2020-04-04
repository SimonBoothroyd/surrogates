import contextlib
import os


@contextlib.contextmanager
def change_directory(path, create=True):
    """A context manager to temporaily change the working
    directory to a specified path, optionally creating the
    directory if it does not exist.
    """

    if create:
        os.makedirs(path, exist_ok=True)

    current_directory = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(current_directory)
