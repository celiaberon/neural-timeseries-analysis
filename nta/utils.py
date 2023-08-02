import functools


def repeat_and_store(num_reps):
    
    '''
    Decorator to repeat function call and stores each call's output in a list.

    Args:
        num_reps:
            Number of times to call inner function.
    
    Returns:
        multi_output:
            List of len()=num_reps with output from inner function.
    '''

    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            multi_output=[]
            for rep in range(num_reps):
                single_output = func(seed=rep, *args, **kwargs)
                multi_output.append(single_output)
            return multi_output
        return wrapper_repeat
    return decorator_repeat