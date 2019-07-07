# simple variable elimination
import numpy as np
from ..factors import Factor, Factors


# P(a)
f1 = Factor(["a"], np.array([0.8, 0.2]))

# P(b|a)
f2 = Factor(["a", "b"], np.array([
    [0.7, 0.3],
    [0.1, 0.9]
]))

# P(c|b)
f3 = Factor(["b", "c"], np.array([
    [0.7, 0.3],
    [0.1, 0.9]
]))

# P(d|c)
f4 = Factor(["c", "d"], np.array([
    [0.7, 0.3],
    [0.1, 0.9]
]))


f = Factors([f1, f2, f3, f4])
result = f.eliminate(["a", "b", "c"])
print(result)