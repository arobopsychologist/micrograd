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

def test_value_add_associative():
    a = Value(1.0)
    b = Value(2.0)
    c = Value(3.0)

    assert (a + b) + c == a + (b + c)

def test_value_add_commutative():
    a = Value(1.0)
    b = Value(2.0)

    assert a + b == b + a

def test_value_mult_associative():
    a = Value(1.0)
    b = Value(2.0)
    c = Value(3.0)

    assert (a * b) * c == a * (b * c)

def test_value_mult_commutative():
    a = Value(1.0)
    b = Value(2.0)

    assert a * b == b * a

def test_value_complex_expression():
    a = Value(1.0)
    b = Value(3.5)
    c = Value(10.0)
    expected_value = Value(36.0)

    print(a + b * c)
    assert a + b * c == expected_value
