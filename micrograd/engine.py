import math


class Value:

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward

        return out

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        topologically_ordered_values = []
        visited = set()

        def build_topologically_ordered_list(v):
            # https://en.wikipedia.org/wiki/Topological_sorting
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topologically_ordered_list(child)
                topologically_ordered_values.append(v)

        build_topologically_ordered_list(self)

        self.grad = 1.0

        for node in reversed(topologically_ordered_values):
            node._backward()
