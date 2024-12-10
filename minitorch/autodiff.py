from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # ASSIGN1.1
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if the variable is a leaf."""
        ...

    def is_constant(self) -> bool:
        """Returns True if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Returns the parent variables and their derivatives with respect to this variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # Code copied over from TA solution
    order = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    visit(parent)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leave nodes."""
    # Code copied over from TA solution
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for var in queue:
        d = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            for v, d in var.chain_rule(d):
                if v.is_constant():
                    continue
                derivatives[v.unique_id] = derivatives.get(v.unique_id, 0.0)
                derivatives[v.unique_id] += d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors from the forward pass."""
        return self.saved_values
