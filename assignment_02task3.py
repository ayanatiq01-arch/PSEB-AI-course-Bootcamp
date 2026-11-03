def is_adult(age):

    if age < 0:
        raise ValueError("Age cannot be negative")

    if age >= 18:
        return True
    else:
        return False
