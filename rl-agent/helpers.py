"""Helper functions"""
from pathlib import Path


def is_sequence_of_type(
    param_name, sequence, sequence_type, element_type, max: int = None, min: int = None
):
    """Check that its a sequence containting a certain type of a certain shape

    Args:
        param_name (str): name of param
        sequence (any): sequence to check
        sequence_type (class): type of sequence
        element_type (class): type of elements in sequence
        max (int, optional): Max length of sequence. Defaults to None.
        min (int, optional): Min length of sequence. Defaults to None.

    Raises:
        ValueError: Wrong object type
        ValueError: Wrong sequence type
        ValueError: Wrong size
    """
    if not isinstance(sequence, sequence_type):
        raise ValueError(f"{param_name}: The object must be of type {sequence_type}")
    list_elements_ok, err = _check_list_elements(sequence, element_type)
    if not list_elements_ok:
        raise ValueError(f"{param_name}: {err}")
    list_size_ok, err = _check_length(sequence, max, min)
    if not list_size_ok:
        raise ValueError(f"{param_name}: {err}")


def is_int(param_name, param, max: int = None, min: int = None):
    if not isinstance(param, int):
        raise ValueError(f"{param_name}: The object must be of type {int}")
    if max is not None and param > max or min is not None and param < min:
        raise ValueError(f"{param_name}: Must be in the interval [{min}, {max}]")


def _check_list_elements(elements, element_type) -> tuple[bool, str]:
    if not all(isinstance(n, element_type) for n in elements):
        return False, f"The tuple must contain elements of type {element_type}"
    return True, ""


def _check_length(list, max, min) -> tuple[bool, str]:
    if max is not None and len(list) > max or min is not None and len(list) < min:
        return False, f"Must have more than {max} elements, and less than {min}"
    return True, ""


def getpath(basedir, filename) -> str:
    # Retrieve root folder
    root = Path(__file__).parent.parent
    filename = root / basedir / filename
    return str(filename)
