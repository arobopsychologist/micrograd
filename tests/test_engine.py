from micrograd.engine import Value
import pytest


def test_eq():
    assert Value(2.0) == Value(2.0)


def test_neq():
    assert Value(2.0) != Value(3.0)


def test_add():
    assert Value(2.0) + Value(-3.0) == Value(-1.0)


def test_add_int():
    assert Value(2.0) + 1 == Value(3.0)


def test_radd_int():
    assert 1 + Value(2.0) == Value(3.0)


def test_mult():
    assert Value(2.0) * Value(-3.0) == Value(-6.0)


def test_mult_int():
    assert Value(2.0) * 2 == Value(4.0)


def test_rmult_int():
    assert 2 * Value(2.0) == Value(4.0)


def test_add_associative():
    a = Value(1.0)
    b = Value(2.0)
    c = Value(3.0)

    assert (a + b) + c == a + (b + c)


def test_add_commutative():
    a = Value(1.0)
    b = Value(2.0)

    assert a + b == b + a


def test_mult_associative():
    a = Value(1.0)
    b = Value(2.0)
    c = Value(3.0)

    assert (a * b) * c == a * (b * c)


def test_mult_commutative():
    a = Value(1.0)
    b = Value(2.0)

    assert a * b == b * a


def test_complex_expression():
    a = Value(1.0)
    b = Value(3.5)
    c = Value(10.0)
    expected_value = Value(36.0)

    assert expected_value == a + b * c


def test_backwards_traversal():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(10.0)
    expected_value = {a * b, c}

    d = a * b + c
    d_prev = d._prev

    assert expected_value == d_prev


def test_op():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(10.0)
    expected_value = "+"

    d = a * b + c
    d_op = d._op

    assert expected_value == d_op


@pytest.mark.parametrize(
    "a, expected_value",
    [
        (Value(-1), Value(-0.7615941559557649)),
        (Value(0), Value(0)),
        (Value(1), Value(0.7615941559557649)),
    ],
)
def test_tanh(a, expected_value):
    assert a.tanh() == expected_value


def test_backpropogation():
    a = Value(1.0)
    b = Value(2.0)
    c = Value(3.0)
    d = Value(4.0)

    e = a + b
    f = c * d

    g = e * f

    g.tanh()

    g.backward()

    assert a.grad == 12.0


def test_backpropogation_with_same_value():
    a = Value(1.0)
    b = a + a
    b.backward()

    assert a.grad == 2.0
