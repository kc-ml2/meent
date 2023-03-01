import jax


class A:
    def __init__(self, theta=0.,):
        self.theta = theta // 2

    def _tree_flatten(self):
        children = (self.theta, )
        aux_data = {}
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @jax.jit
    def solve_jit(self):
        return self.theta

    def solve_no_jit(self):
        return self.theta


jax.tree_util.register_pytree_node(A, A._tree_flatten, A._tree_unflatten)

theta = 4
print(theta, ': theta original')

solver = A(theta=theta)
print(solver.theta, ': theta after __init__')

# a = emsolver.solve_no_jit()
# b = jax.jit(emsolver.solve_no_jit)()
c = solver.solve_jit()
solver.theta = 8
d = solver.solve_jit()

# print(a, ': theta from no-jit', )
# print(b, ': theta from outside_jit', )
print(c, ': theta from inside-jit', )
print(d, ': theta from inside-jit', )
