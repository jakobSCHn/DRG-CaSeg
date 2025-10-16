import attrs

def validate_uint8(instance, attribute, value):
    """
    Validator to ensure a value is an integer between 0 and 255.
    """
    if not isinstance(value, int) or not (0 <= value <= 255):
        raise ValueError(
            f"'{attribute.name}' must be an integer between 0 and 255, "
            f"but got {value}"
        )

def validate_positive(instance, attribute, value):
    """
    Validator to ensure a value is a positive number.
    """
    if not (value > 0):
        raise ValueError(
            f"'{attribute.name}' must be positive, but got {value}"
        )