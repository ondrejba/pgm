# student network from the PGM book (page 53)
import copy as cp
import numpy as np
from ..factors import Factor, Factors


# factor names
DIFFICULTY = "d"
INTELLIGENCE = "i"
GRADE = "g"
SAT = "s"
LETTER = "l"

NAMES = [DIFFICULTY, INTELLIGENCE, GRADE, SAT, LETTER]
assert len(set(NAMES)) == len(NAMES)

# factors
difficulty = Factor([DIFFICULTY], np.array([0.6, 0.4]))

intelligence = Factor([INTELLIGENCE], np.array([0.7, 0.3]))

grade = Factor([DIFFICULTY, INTELLIGENCE, GRADE], np.array([
    [
        [0.3, 0.4, 0.3],
        [0.05, 0.25, 0.7]
    ],
    [
        [0.9, 0.08, 0.02],
        [0.5, 0.3, 0.2]
    ]
]))

sat = Factor([INTELLIGENCE, SAT], np.array([
    [0.95, 0.05],
    [0.2, 0.8]
]))

letter = Factor([GRADE, LETTER], np.array([
    [0.1, 0.9],
    [0.4, 0.6],
    [0.99, 0.01]
]))

f = Factors([difficulty, intelligence, grade, sat, letter])

# inference
print("Joint probability of letter (good/bad) and SAT (low/high) without evidence:")
tmp_f = cp.deepcopy(f)
result = tmp_f.eliminate([DIFFICULTY, INTELLIGENCE, GRADE])
print(result)
print()

# conditioning + inference
print("Probability of intelligence (low/high) given a good letter:")
tmp_f = cp.deepcopy(f)
tmp_f.condition([LETTER], [0])
result = tmp_f.eliminate([DIFFICULTY, GRADE, SAT])
result.normalize()
print(result)
print()

print("Probability of intelligence (low/high) given a good letter and a high SAT score:")
tmp_f = cp.deepcopy(f)
tmp_f.condition([LETTER, SAT], [0, 1])
result = tmp_f.eliminate([DIFFICULTY, GRADE])
result.normalize()
print(result)


