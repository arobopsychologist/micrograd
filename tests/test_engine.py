from micrograd.engine import Value

def test_value_eq():
    a = Value(2.0)
    b = Value(2.0)

    assert a == b

def test_value_neq():
    a = Value(2.0)
    b = Value(3.0)

    assert a != b

def test_value_add():
    a = Value(2.0)
    b = Value(-3.0)
    expected_sum = Value(-1.0)

    assert a + b == expected_sum

def test_value_mult():
    a = Value(2.0)
    b = Value(-3.0)
    expected_product = Value(-6.0)

    assert a * b == expected_product
