from sin21.src.hw2.hw2 import Num
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from src.hw2 import hw2
import pytest


def test_Sample():
    smpl = hw2.Sample(0)
    assert smpl.oid == 0, "test failed"


def test_Num():
    num = hw2.Num()
    assert num.n == 0, "test failed"


def test_Sym():
    sym = hw2.Sym()
    assert sym.n == 0, "test failed"


def test_Col():
    col = hw2.Col(0, 'e', hw2.Sym(),True)
    assert col.oid == 0, "test failed"
