'''
# -----------------
#   util_calc
# -----------------

'''

# == imports ==
# -- packages --
import numpy as np

# -- imported scripts --



# == calculate mlr ==
def standardize_variable(x):
    return (x - np.mean(x)) / np.std(x)


def calculate_mlr(y, x_list):
    X = np.column_stack([np.ones(len(y))] + x_list)
    b = np.linalg.inv(X.T @ X) @ X.T @ y

    print(b)
    exit()
    
    return b


















































