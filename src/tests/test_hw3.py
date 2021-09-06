import hw3


def test_Sample():
    smpl = hw3.Sample(0)
    assert smpl.oid == 0, "test failed"


def test_Num():
    num = hw3.Num()
    assert num.n == 0, "test failed"


def test_Sym():
    sym = hw3.Sym()
    assert sym.n == 0, "test failed"


def test_Col():
    col = hw3.Col(0, 'e', hw3.Sym(),True)
    assert col.oid == 0, "test failed"

def test_Clone():
    smpl = hw3.Sample(0)
    smpl2 = smpl.clone()
    assert smpl.oid == smpl2.oid, "test failed"
