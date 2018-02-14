import numpy as np

def unstability(x):
    '''
    Calculates a measure of the unstability of the learning. More precisely the unstability
    is determined as a sum of the relative increases of the function value over the iterations.
    Refer to the attached report for a more rigorous definition.
    :param x: A 1-dimensional array representing the function value at each iteration.
    :return: The value of the unstability
    '''
    return \
        np.sum(0 if x[i] <= x[i - 1] else np.abs(x[i - 1] - x[i]) / x[i - 1] for i in range(1, len(x)))


def conv_speed(x, eps=10):
    '''
    Calculates a measure of convergence speed. More precisely it is defined in the [0,100] range.
    A value of 100 is achieved when the function' value is within a percent eps of the minimum
    value reached at the first iteration. A value of 0 is achieved when the function' value
    is not within the range of the minimum value until the last iteration.
    :param x: A 1-dimensional array representing the function value at each iteration.
    :param eps: Specifies a the desired distance from the minimum value as a percentage.
    :return: The value of the convergence speed.
    '''
    return 100 - np.argmax(x <= np.min(x) * (1 + eps / 100)) * 100 / (len(x) - 1)
