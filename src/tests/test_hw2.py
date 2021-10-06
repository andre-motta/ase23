import hw2


def test_Sample():
    smpl = hw2.Sample(0)
    assert smpl.oid == 0, "test failed"


def test_Num():
    num = hw2.Num(0, 'e')
    assert num.n == 0, "test failed"


def test_Sym():
    sym = hw2.Sym(0, 'e')
    assert sym.n == 0, "test failed"


def test_Col():
    col = hw2.Col(0, 'e', hw2.Sym(0, 'e'),True)
    assert col.oid == 0, "test failed"
