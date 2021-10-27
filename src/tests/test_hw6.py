import hw6


def test_Sample():
    smpl = hw6.Sample(0)
    assert smpl.oid == 0, "test failed"


def test_Num():
    num = hw6.Num(0, 'e')
    assert num.n == 0, "test failed"


def test_Sym():
    sym = hw6.Sym(0, 'e')
    assert sym.n == 0, "test failed"


def test_Col():
    col = hw6.Col(0,'e', hw6.Sym(0, 'e'),True)
    assert col.oid == 0, "test failed"

def test_Clone():
    smpl = hw6.Sample(0)
    smpl2 = smpl.clone()
    assert smpl.oid == smpl2.oid, "test failed"
