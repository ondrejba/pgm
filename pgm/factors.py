from functools import reduce
import itertools
import numpy as np


class Factor:

    def __init__(self, vars, table):

        assert len(vars) == len(table.shape)

        self.vars = vars
        self.table = table

    def reduce(self, var, value_idx):

        index = self.vars.index(var)
        self.table = np.take(self.table, value_idx, axis=index)
        del self.vars[index]

    def has_var(self, var):
        return var in self.vars

    def var_index(self, variable):
        return self.vars.index(variable)

    def sum_over(self, var):

        var_idx = self.vars.index(var)
        del self.vars[var_idx]
        self.table = np.sum(self.table, axis=var_idx)

    def is_valid(self):
        return len(self.vars) > 0

    def normalize(self):
        self.table = self.table / np.sum(self.table)

    def __mul__(self, factor):

        vars_1 = set(self.vars)
        vars_2 = set(factor.vars)
        all_variables = list(sorted(vars_1.union(vars_2)))

        index_1 = []
        index_2 = []
        mask_1 = []
        mask_2 = []
        num_values = []

        for var in all_variables:

            if self.has_var(var):
                index_1.append(self.var_index(var))
                mask_1.append(True)
                num_values.append(self.table.shape[index_1[-1]])
            else:
                mask_1.append(False)

            if factor.has_var(var):
                index_2.append(factor.var_index(var))
                mask_2.append(True)

                if not mask_1[-1]:
                    num_values.append(factor.table.shape[index_2[-1]])
            else:
                mask_2.append(False)

        mask_1 = np.array(mask_1)
        mask_2 = np.array(mask_2)

        new_table = np.empty(num_values, dtype=np.float32)

        for assignment in itertools.product(*[range(x) for x in num_values]):

            assignment_np = np.array(assignment)
            assignment_1 = list(assignment_np[mask_1][index_1])
            assignment_2 = list(assignment_np[mask_2][index_2])

            val = self.table[tuple(assignment_1)] * factor.table[tuple(assignment_2)]
            new_table[tuple(assignment)] = val

        return Factor(all_variables, new_table)

    def __rmul__(self, factor):

        return self.__mul__(factor)

    def __str__(self):

        str = "variables: {}\n".format(self.vars)
        str += "table:\n"
        str += np.array2string(self.table)
        str += "\nsum: {:.8f}".format(np.sum(self.table))

        return str


class Factors:

    def __init__(self, factors):

        self.factors = factors

    def condition(self, vars, values):

        assert len(vars) == len(values)

        to_delete = []

        for factor in self.factors:

            for var, value in zip(vars, values):
                if factor.has_var(var):
                    factor.reduce(var, value)

            if not factor.is_valid():
                to_delete.append(factor)

        for factor in to_delete:

            self.factors.remove(factor)

    def eliminate(self, ordering):

        for var in ordering:
            self.eliminate_variable(var)

        assert len(self.factors) > 0

        return reduce((lambda x, y: x * y), self.factors)

    def eliminate_variable(self, var):

        relevant_factors = []
        other_factors = []

        for factor in self.factors:
            if factor.has_var(var):
                relevant_factors.append(factor)
            else:
                other_factors.append(factor)

        new_factor = reduce((lambda x, y: x * y), relevant_factors)
        new_factor.sum_over(var)

        self.factors = other_factors + [new_factor]
