def check_positive(var, var_name, strictly=True):
    """Raise an error if variable is not positive.

    Parameters
    ----------
    var : unspecified
        Variable to be tested.
    var_name : string
        Name of the variable.
    strictly : bool
        Check if var is strictly positve or not.
    """
    if strictly:
        condition = var > 0
    else:
        condition = var >= 0

    if not condition:
        raise ValueError(
            'Variable {} needs to be positive, got: {}'.format(
                var_name, var))


def check_type(var, var_name, accepted_types):
    """Raise an error if variable does not have an accepted type.

    Parameters
    ----------
    var : unspecified
        Variable to be tested.
    var_name : string
        Name of the variable.
    accepted_types : list
        Accepted types that the variable can take.
    """
    if type(var) not in accepted_types:
        raise TypeError(
            'Variable\'s type {} needs to be in {}, got: {}'.format(
                var_name, accepted_types, type(var)))


def check_callable(var, var_name):
    """Raise an error if variable is not callable.

    Parameters
    ----------
    var : unspecified
        Variable to be tested.
    var_name : string
        Name of the variable.
    """
    if callable(var) is False:
        raise TypeError(
            'Variable {} needs to be callable.'.format(var_name))


def check_value(var, var_name, accepted_values):
    """Raise an error if var does not belong to a set of values.

    Parameters
    ----------
    var : unspecified
        Parameter to be tested.
    var_name : string
        Name of the variable.
    accepted_values : list
        Accepted values that the variable can take.
    """
    if var not in accepted_values:
        raise ValueError(
            'Variable {} needs to be in {}, got: {}'.format(
                var_name, accepted_values, var))
