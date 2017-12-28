from autodiff import function, gradient

# -- use Theano to compile a function

@function
def f(x):
    return x ** 2

print(f(5.0)) # returns 25.0; not surprising but executed in Theano!

# -- automatically differentiate a function with respect to its inputs

@gradient
def f(x):
    return x ** 2

print(f(5.0))