import math
import re
from collections import defaultdict
import random
from operator import itemgetter
import copy
from statistics import median
r = random.random
seed = random.seed
from random import sample
from functools import cmp_to_key

"""
This code reads allows users to read in a csv file, 
and retain information on classes, weights, numbers, 
symbols and goals.

HW4 & 5
Task 1:
|.. n=398 c= 0.94
|.. |.. n=199 c= 0.64
|.. |.. |.. n=99 c= 0.64
|.. |.. |.. |.. n=49 c= 0.32
|.. |.. |.. |.. |.. n=24 c= 0.25
|.. |.. |.. |.. |.. |.. n=12 c= 0.24     goals = [ 2419.5, 15.2, 30.0]
|.. |.. |.. |.. |.. |.. n=12 c= 0.20     goals = [ 1985.0, 17.1, 30.0]
|.. |.. |.. |.. |.. n=25 c= 0.23
|.. |.. |.. |.. |.. |.. n=12 c= 0.20     goals = [ 2177.5, 16.9, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.06     goals = [ 1985.0, 16.9, 40.0]
|.. |.. |.. |.. n=50 c= 0.49
|.. |.. |.. |.. |.. n=25 c= 0.50
|.. |.. |.. |.. |.. |.. n=12 c= 0.47     goals = [ 2283.0, 14.8, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.18     goals = [ 2124.0, 17.0, 30.0]
|.. |.. |.. |.. |.. n=25 c= 0.20
|.. |.. |.. |.. |.. |.. n=12 c= 0.13     goals = [ 2156.0, 18.5, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.17     goals = [ 2219.0, 15.5, 30.0]
|.. |.. |.. n=100 c= 0.65
|.. |.. |.. |.. n=50 c= 0.54
|.. |.. |.. |.. |.. n=25 c= 0.54
|.. |.. |.. |.. |.. |.. n=12 c= 0.54     goals = [ 2207.5, 14.9, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.20     goals = [ 2125.0, 15.9, 30.0]
|.. |.. |.. |.. |.. n=25 c= 0.17
|.. |.. |.. |.. |.. |.. n=12 c= 0.15     goals = [ 2652.5, 15.8, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.05     goals = [ 2625.0, 17.3, 30.0]
|.. |.. |.. |.. n=50 c= 0.65
|.. |.. |.. |.. |.. n=25 c= 0.57
|.. |.. |.. |.. |.. |.. n=12 c= 0.55     goals = [ 2249.5, 17.6, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.45     goals = [ 2500.0, 15.8, 30.0]
|.. |.. |.. |.. |.. n=25 c= 0.28
|.. |.. |.. |.. |.. |.. n=12 c= 0.22     goals = [ 2849.0, 15.6, 20.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.12     goals = [ 1990.0, 14.9, 30.0]
|.. |.. n=199 c= 0.70
|.. |.. |.. n=99 c= 0.58
|.. |.. |.. |.. n=49 c= 0.56
|.. |.. |.. |.. |.. n=24 c= 0.52
|.. |.. |.. |.. |.. |.. n=12 c= 0.46     goals = [ 3070.5, 16.2, 20.0]
|.. |.. |.. |.. |.. |.. n=12 c= 0.38     goals = [ 3208.5, 15.8, 20.0]
|.. |.. |.. |.. |.. n=25 c= 0.40
|.. |.. |.. |.. |.. |.. n=12 c= 0.39     goals = [ 2245.0, 16.5, 25.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.28     goals = [ 2634.0, 15.5, 20.0]
|.. |.. |.. |.. n=50 c= 0.40
|.. |.. |.. |.. |.. n=25 c= 0.32
|.. |.. |.. |.. |.. |.. n=12 c= 0.24     goals = [ 2860.0, 15.9, 20.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.25     goals = [ 3245.0, 16.6, 20.0]
|.. |.. |.. |.. |.. n=25 c= 0.22
|.. |.. |.. |.. |.. |.. n=12 c= 0.22     goals = [ 3472.5, 16.6, 20.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.10     goals = [ 3264.0, 17.8, 20.0]
|.. |.. |.. n=100 c= 0.52
|.. |.. |.. |.. n=50 c= 0.29
|.. |.. |.. |.. |.. n=25 c= 0.23
|.. |.. |.. |.. |.. |.. n=12 c= 0.21     goals = [ 4365.0, 10.0, 10.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.16     goals = [ 4668.0, 11.5, 10.0]
|.. |.. |.. |.. |.. n=25 c= 0.19
|.. |.. |.. |.. |.. |.. n=12 c= 0.15     goals = [ 3792.5, 12.0, 15.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.14     goals = [ 4274.0, 13.0, 10.0]
|.. |.. |.. |.. n=50 c= 0.35
|.. |.. |.. |.. |.. n=25 c= 0.30
|.. |.. |.. |.. |.. |.. n=12 c= 0.16     goals = [ 4110.0, 13.2, 20.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.20     goals = [ 3725.0, 15.0, 20.0]
|.. |.. |.. |.. |.. n=25 c= 0.20
|.. |.. |.. |.. |.. |.. n=12 c= 0.13     goals = [ 4215.0, 13.3, 10.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.11     goals = [ 4098.0, 14.0, 10.0]
|.. n=398 c= 0.91
|.. |.. n=199 c= 0.73
|.. |.. |.. n=99 c= 0.56
|.. |.. |.. |.. n=49 c= 0.56
|.. |.. |.. |.. |.. n=24 c= 0.26
|.. |.. |.. |.. |.. |.. n=12 c= 0.15     goals = [ 1985.0, 16.9, 30.0]
|.. |.. |.. |.. |.. |.. n=12 c= 0.18     goals = [ 2283.5, 16.5, 25.0]
|.. |.. |.. |.. |.. n=25 c= 0.46
|.. |.. |.. |.. |.. |.. n=12 c= 0.47     goals = [ 2130.0, 15.2, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.13     goals = [ 2511.0, 15.5, 20.0]
|.. |.. |.. |.. n=50 c= 0.34
|.. |.. |.. |.. |.. n=25 c= 0.25
|.. |.. |.. |.. |.. |.. n=12 c= 0.25     goals = [ 3045.0, 15.9, 25.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.16     goals = [ 2130.0, 15.8, 40.0]
|.. |.. |.. |.. |.. n=25 c= 0.19
|.. |.. |.. |.. |.. |.. n=12 c= 0.17     goals = [ 1945.0, 15.4, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.13     goals = [ 2464.0, 15.5, 20.0]
|.. |.. |.. n=100 c= 0.64
|.. |.. |.. |.. n=50 c= 0.65
|.. |.. |.. |.. |.. n=25 c= 0.60
|.. |.. |.. |.. |.. |.. n=12 c= 0.59     goals = [ 2640.0, 14.6, 25.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.14     goals = [ 2350.0, 15.0, 30.0]
|.. |.. |.. |.. |.. n=25 c= 0.50
|.. |.. |.. |.. |.. |.. n=12 c= 0.47     goals = [ 2052.5, 18.0, 35.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.08     goals = [ 1995.0, 16.9, 30.0]
|.. |.. |.. |.. n=50 c= 0.47
|.. |.. |.. |.. |.. n=25 c= 0.45
|.. |.. |.. |.. |.. |.. n=12 c= 0.36     goals = [ 2659.0, 16.4, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.37     goals = [ 2625.0, 16.4, 30.0]
|.. |.. |.. |.. |.. n=25 c= 0.23
|.. |.. |.. |.. |.. |.. n=12 c= 0.13     goals = [ 2137.5, 15.2, 30.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.15     goals = [ 2230.0, 15.9, 30.0]
|.. |.. n=199 c= 0.70
|.. |.. |.. n=99 c= 0.58
|.. |.. |.. |.. n=49 c= 0.56
|.. |.. |.. |.. |.. n=24 c= 0.52
|.. |.. |.. |.. |.. |.. n=12 c= 0.52     goals = [ 3061.5, 15.8, 20.0]
|.. |.. |.. |.. |.. |.. n=12 c= 0.08     goals = [ 3199.5, 16.5, 20.0]
|.. |.. |.. |.. |.. n=25 c= 0.40
|.. |.. |.. |.. |.. |.. n=12 c= 0.39     goals = [ 2245.0, 16.5, 25.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.28     goals = [ 2634.0, 15.5, 20.0]
|.. |.. |.. |.. n=50 c= 0.40
|.. |.. |.. |.. |.. n=25 c= 0.26
|.. |.. |.. |.. |.. |.. n=12 c= 0.13     goals = [ 3002.5, 16.5, 20.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.26     goals = [ 3410.0, 17.2, 20.0]
|.. |.. |.. |.. |.. n=25 c= 0.25
|.. |.. |.. |.. |.. |.. n=12 c= 0.24     goals = [ 3041.0, 17.1, 20.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.08     goals = [ 3459.0, 16.9, 20.0]
|.. |.. |.. n=100 c= 0.52
|.. |.. |.. |.. n=50 c= 0.28
|.. |.. |.. |.. |.. n=25 c= 0.21
|.. |.. |.. |.. |.. |.. n=12 c= 0.19     goals = [ 4423.5, 10.5, 10.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.15     goals = [ 4385.0, 12.0, 10.0]
|.. |.. |.. |.. |.. n=25 c= 0.16
|.. |.. |.. |.. |.. |.. n=12 c= 0.14     goals = [ 4328.5, 13.0, 10.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.15     goals = [ 3693.0, 12.0, 20.0]
|.. |.. |.. |.. n=50 c= 0.34
|.. |.. |.. |.. |.. n=25 c= 0.22
|.. |.. |.. |.. |.. |.. n=12 c= 0.14     goals = [ 4070.0, 13.8, 10.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.11     goals = [ 4380.0, 13.2, 10.0]
|.. |.. |.. |.. |.. n=25 c= 0.26
|.. |.. |.. |.. |.. |.. n=12 c= 0.21     goals = [ 4097.5, 13.1, 20.0]
|.. |.. |.. |.. |.. |.. n=13 c= 0.17     goals = [ 3830.0, 14.3, 20.0]

Task 2:
Weight-     Acceleration+     Mpg+     
2052.5      18.0              35.0      <== best
2130.0      15.8              40.0
1985.0      16.9              30.0
1995.0      16.9              30.0
1945.0      15.4              30.0
2130.0      15.2              30.0
2137.5      15.2              30.0
2230.0      15.9              30.0
2350.0      15.0              30.0
2625.0      16.4              30.0
2659.0      16.4              30.0
2245.0      16.5              25.0
2283.5      16.5              25.0
2640.0      14.6              25.0
2464.0      15.5              20.0
2511.0      15.5              20.0
3045.0      15.9              25.0
3041.0      17.1              20.0
2634.0      15.5              20.0
3002.5      16.5              20.0
3199.5      16.5              20.0
3410.0      17.2              20.0
3061.5      15.8              20.0
3459.0      16.9              20.0
3830.0      14.3              20.0
3693.0      12.0              20.0
4097.5      13.1              20.0
4070.0      13.8              10.0
4328.5      13.0              10.0
4380.0      13.2              10.0
4385.0      12.0              10.0
4423.5      10.5              10.0      <== worst

Task 3:
{at:1, txt:Displacement, lo:85, hi:91, best:9, rest:0}
{at:1, txt:Displacement, lo:97, hi:400, best:3, rest:3}
{at:1, txt:Displacement, lo:429, hi:455, best:0, rest:9}

{at:2, txt:Horsepower, lo:52, hi:65, best:7, rest:0}
{at:2, txt:Horsepower, lo:67, hi:97, best:5, rest:0}
{at:2, txt:Horsepower, lo:167, hi:208, best:0, rest:5}
{at:2, txt:Horsepower, lo:215, hi:230, best:0, rest:7}

{at:6, txt:origin, lo:1, hi:1, best:1, rest:12}
{at:6, txt:origin, lo:3, hi:3, best:11, rest:0}

Task 4:
Best
best ['4', '85', '52', '2035', '22.2', '76', '1', '30']
best ['4', '119', '97', '2545', '17', '75', '3', '20']
best ['4', '97', '75', '2265', '18.2', '77', '3', '30']
best ['4', '98', '68', '2135', '16.6', '78', '3', '30']
best ['4', '85', '70', '2070', '18.6', '78', '3', '40']
best ['4', '85', '65', '2020', '19.2', '79', '3', '30']

worst ['8', '455', '225', '4951', '11', '73', '1', '10']
worst ['8', '440', '215', '4735', '11', '73', '1', '10']
worst ['8', '429', '198', '4952', '11.5', '73', '1', '10']
worst ['8', '400', '230', '4278', '9.5', '73', '1', '20']
worst ['8', '429', '208', '4633', '11', '72', '1', '10']
worst ['8', '455', '225', '4425', '10', '70', '1', '10']

"""

#Based on class o by Tim Menzies
class Bag:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        return "{"+ ', '.join([f"{k}:{v}" for k, v in list(self.__dict__.items())[:-2] if  k[0] != "_"]) + "}"
    
    def __getitem__(self, key):
        return self.__dict__.get(key)


def park_miller(seed,start,end):
    """
    Util random number generator park-miller method.
    """
    a = (end-start)/2147483647
    b = start
    while True:
        seed = (16807*seed) % 2147483647
        yield  a * seed +b

def var(items):
    n = len(items)
    return items[ 9*(n // 10) ] - items[ 9*(n // 10) ] / 2.56

def unsuper(data, min_break, min_size):
    data.sort(key=lambda d: d[0])
    groups = []
    group = []
    nums = []

    while len(data) > 0:
        d = data.pop(0)
        x, _ = d

        group.append(d)
        nums.append(x)

        x1 = None if len(data) == 0 else data[0][0]

        if (x1 is None or (x != x1)) and (nums[-1] - nums[0] > min_break) and (len(nums) >= min_size):
            groups.append(group)
            group = []
            nums = []
    
    if len(group) > 0:
        if len(groups) > 0:
            groups[-1] += group
        else:
            groups = [group]
    
    return groups

def merge(groups):
    proposal = []
    i = 0
    while i < len(groups) - 1:
        current = [g[0] for g in groups[i]]
        next = [g[0] for g in groups[i + 1]]
        both = current + next

        n1, n2 = len(current), len(next)
        var1, var2, var3 = var(current), var(next), var(both)

        if var3*0.95 <= (var1*n1 + var2*n2)/(n1+n2):
            proposal.append( groups[i] + groups[i+1] )
            i += 2
        else:
            proposal.append(groups[i])
            i += 1
    
    if i == len(groups) - 1:
        proposal.append(groups[i])

    if len(proposal) < len(groups):
        return merge(proposal)
    else:
        return groups

class Sym:
    """
    Sym class represents the symbolic information
    """
    

    def __init__(self,oid,txt, data=None):
        """
        construct a new Sym object
        """
        self.n = 0
        self.most = 0
        self.mode = ""
        self.oid = oid
        self.txt = txt
        self.cnt = defaultdict(int)
        if data != None:
            for val in data:
                self + val

    def __add__(self, v):
        """
        add method for Sym
        """
        self.n += 1
        self.cnt[v] += 1
        tmp = self.cnt[v]
        if tmp > self.most:
            self.most = tmp
            self.mode = v
        return v

    def __sub__(self, x):
        """
        sub method for Sym
        """
        old = self.cnt.get(x, 0)
        if old > 0:
            self.cnt[x] = old - 1

    def variety(self):
        """
        returns entropy
        """
        return self.syment()

    def xpect(self, j):
        """
        expectation for Sym
        """
        n = self.n + j.n
        return self.n / n * self.variety() + j.n / n * j.variety()

    def syment(self):
        """
        entropy for Sym
        """
        e = 0
        for k, v in self.cnt.items():
            p = v/self.n
            e -= p*math.log(p)/math.log(2)
        return e

    def symany(self, without):
        r = random.randint()
        for k, v in self.cnt.items():
            m = self.n - v if without else v
            r -= m/self.n
            if r <= 0:
                return k
        return k

    def symlike(self,x,prior,m):
        """
        calculates likelihood between syms
        """
        f = self.cnt[x]
        return (f + m*prior)/(self.n + m)

    def dist(self,x,y):
        if (x == "?") or (y == "?"):
            return 1
        return 0 if x == y else 1

    def get(self, key):
        x = self.cnt.get(key)
        if x is None:
            x = 0
        return x

    def discretize(self, other):
        for val in set(self.cnt.keys() | other.cnt.keys()):
            yield Bag( at = self.oid-1, txt = self.txt, lo = val, hi = val,
                    best = self.get(val),
                    rest = other.get(val),
                    first = False, last = False )

class Num:
    """
    Num class represents the numeric information
    """
    

    def __init__(self, oid, txt, data=None):
        """
        construct a new Num object
        """
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.sd = 0
        self.lo = float('inf')
        self.hi = -float('inf')
        self.val = []
        self.oid = oid
        self.txt = txt
        if data != None:
            for val in data:
                self + val
                self.val.append(val)
        
    def variety(self):
        """
        returns standard deviation
        """
        return self.sd

    def xpect(self, j):
        """
        expectation for Num
        """
        n = self.n + j.n
        return self.n / n * self.variety() + j.n / n * j.variety()

    def _numSd(self):
        """
        standard deviation of Num
        """
        if self.m2 < 0:
            return 0
        if self.n < 2:
            return 0
        return math.sqrt(self.m2/(self.n - 1))

    def numNorm(self, x):
        """
        normalization to Num
        """
        return (x - self.lo)/(self.hi - self.lo + 10e-32)

    def __sub__(self, v):
        """
        sub method for Sym
        """
        if self.n < 2:
            self.sd = 0
            return v
        self.n -= 1
        d = v - self.mu
        self.mu -= d / self.n
        self.m2 -= d*(v - self.mu)
        self.sd = self._numSd()
        return v

    def __add__(self, v):
        """
        add method for Sym
        """
        self.n += 1
        self.val.append(v)
        if v < self.lo:
            self.lo = v
        if v > self.hi:
            self.hi = v
        d = v - self.mu
        self.mu += d / self.n
        self.m2 += d * (v - self.mu)
        self.sd = self._numSd()
        return v

    def numlike(self, x):
        """
        calculates likelihood between Nums
        """
        var = self.sd**2
        denom = math.sqrt(math.pi * 2 * var)
        num = (2.71828**(-(x-self.mu)**2)/(2*var+0.0001))
        return num/(denom+10**(-64)) + 10**(-64)
    
    def dist(self, x, y):
        if (x == "?") and (y == "?"):
            return 1
        if (x == "?") or (y == "?"):
            x = x if (y == "?") else y
            x = self.numNorm(x)
            y = 0 if x > 0.5 else 1
            return y - x
        return self.numNorm(x) - self.numNorm(y)

    def discretize(self, other):
        cohen = 0.3
        # Organize data
        X = [(good, 1) for good in self.val] + [(bad, 0) for bad in other.val]
        n1 = self.n
        n2 = other.n
        iota = cohen * (self.sd*n1 + other.sd*n2) / (n1 + n2)
        clusters = unsuper(X, iota, math.sqrt(len(X)))
        ranges = merge(clusters)
        
        if len(ranges) > 1:
            for n, r in enumerate(ranges):
                counts = [x[1] for x in r]
                yield Bag( at = self.oid-1, txt = self.txt, lo = r[0][0], hi = r[-1][0],
                        best = sum(1 for x in counts if x == 1), 
                        rest = sum(1 for x in counts if x == 0),
                        first = (n == 0), last = (n == len(ranges)))

class Col:
    """
    Col class keeps information on the columns
    """
    def __init__(self, oid, txt, obj, isnum):
        """
        construct a new Col object
        """
        self.oid, self.txt, self.obj, self.isnum = oid, txt, obj, isnum

    def __add__(self, v):
        """
        add method for Col
        """
        self.obj + v

    def __sub__(self, v):
        """
        sub method for Col
        """
        self.obj - v

    def variety(self):
        """
        returns variety of Col
        """
        return self.obj.variety()

    def dist(self, x, y):
        """
        returns distance between two objects for Col
        """
        return self.obj.dist(x,y)

    def discretize(self, other):
        return self.obj.discretize(other.obj)

class Sample:
    """
    Sample class keeps a bundle of columns
    """
    def __init__(self, oid):
        """
        construct a new Sample object
        """
        self.oid = oid
        self.count = 0
        self.cols = []
        self.rows = []
        self.klass = -1
        self.fileline = 0
        self.linesize = 0
        self.skip = []
        self.y = []
        self.nums = []
        self.syms = []
        self.w = defaultdict(int)
        self.xnums = []
        self.x = []
        self.xsyms = []
        self.header = ""

    @staticmethod
    def read(file):
        """
        file reader
        """
        lines = []
        with open(file) as f:
            curline = ""
            for line in f:
                line = line.strip()
                if line[len(line) -1 ] ==',':
                    curline += line
                else:
                    curline+= line
                    lines.append(curline)
                    curline = ""
        return lines

    @staticmethod
    def linemaker(src, sep=",", doomed=r'([\n\t\r ]|#.*)'):
        """
        line maker
        """
        lines = []
        "convert lines into lists, killing whitespace and comments"
        for line in src:
            line = line.strip()
            line = re.sub(doomed, '', line)
            if line:
                lines.append(line.split(sep))
        return lines

    @staticmethod
    def string(s):
        """
        stringify lines
        """
        lines = []
        for line in s.splitlines():
            lines.append(line)
        return lines

    @staticmethod
    def compiler(x):
        """
        automatic type conversion
        """
        try:
            int(x)
            return int
        except:
            try:
                float(x)
                return float
            except ValueError:
                return str

    def zitler(self,row1, row2):
        """
        continuous domination predicate
        """
        goals = self.y
        s1,s2,e,n =0,0,2.71828,len(goals)
        for goal in goals:
            col = self.cols[goal-1]
            w = 1
            if goal in self.w:
                w = -1
            x = col.obj.numNorm(self.convert(row1[goal-1]))
            y = col.obj.numNorm(self.convert(row2[goal-1]))
            s1 = s1 - e**(w * (x-y)/n) # what-if #1: try going x to y
            s2 = s2 - e**(w * (y-x)/n) # what-if #2: try going y to x
        return s1/n < s2/n
    
    def partition(self, arr, low, high):
        """
        partition for quicksort
        """
        i = (low-1)         # index of smaller element
        pivot = arr[high]     # pivot
    
        for j in range(low, high):
    
            # If current element is smaller than or
            # equal to pivot
            if self.zitler(arr[j], pivot):
    
                # increment index of smaller element
                i = i+1
                arr[i], arr[j] = arr[j], arr[i]
    
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return (i+1)
    
    def quickSort(self,arr, low, high):
        """
        quickSort function using zitler's continuous domination predicate
        """
        if len(arr) == 1:
            return arr
        if low < high:
    
            # pi is partitioning index, arr[p] is now
            # at right place
            pi = self.partition(arr, low, high)
    
            # Separately sort elements before
            # partition and after partition
            self.quickSort(arr, low, pi-1)
            self.quickSort(arr, pi+1, high)
    
    def sorted(self):
        self.quickSort(self.rows,0,len(self.rows)-1)
        return self.rows

    def idxPartition(self, arr, low, high):
        """
        partition for quicksort
        """
        i = (low-1)         # index of smaller element
        pivot = arr[high][1]     # pivot
    
        for j in range(low, high):
    
            # If current element is smaller than or
            # equal to pivot
            if self.zitler(arr[j][1], pivot):
    
                # increment index of smaller element
                i = i+1
                arr[i], arr[j] = arr[j], arr[i]
    
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return (i+1)
    
    def idxQuickSort(self,arr, low, high):
        """
        quickSort function using zitler's continuous domination predicate
        """
        if len(arr) == 1:
            return arr
        if low < high:
    
            # pi is partitioning index, arr[p] is now
            # at right place
            pi = self.idxPartition(arr, low, high)
    
            # Separately sort elements before
            # partition and after partition
            self.idxQuickSort(arr, low, pi-1)
            self.idxQuickSort(arr, pi+1, high)

    def convert(self, x):
        """
        automatic type conversion for a Sample
        """
        f = self.compiler(x)
        return f(x)

    def create_cols(self, line):
        """
        creates the Sample header with it's cols
        """
        self.header = line
        index = 0
        cols = []
        for val in line:
            val = self.convert(val)
            if val[0] == "?":
                self.skip.append(index+1)
            if val[0].isupper() or "-" in val or "+" in val:
                self.nums.append(index + 1)
                cols.append(''.join(c for c in val if not c in ['?']))
                self.cols.append(Col(index+1, str(''.join(c for c in val if not c in ['?'])), Num(index+1, str(''.join(c for c in val if not c in ['?']))), True))
            else:
                self.syms.append(index + 1)
                cols.append(''.join(c for c in val if not c in ['?']))
                self.cols.append(Col(index+1, str(''.join(c for c in val if not c in ['?'])), Sym(index+1, str(''.join(c for c in val if not c in ['?']))), False))

            if "!" in val or "-" in val or "+" in val:
                self.y.append(index+1)
                if "-" in val:
                    self.w[index+1] = -1
                if "!" in val:
                    self.klass = index
            if "-" not in val and "+" not in val and "!" not in val:
                self.x.append(index + 1)
                if val[0].isupper():
                    self.xnums.append(index + 1)
                else:
                    self.xsyms.append(index + 1)
            index += 1
        self.linesize = index
        self.fileline += 1

    def insert_row(self, line):
        """
        inserts a row into a Sample
        """
        self.fileline += 1
        if len(line) < self.linesize:
            print("Line", self.fileline, "has an error")
            return
        realline = []
        realindex = 0
        index = 0
        for val in line:
            if index + 1 not in self.skip:
                if val == "?":
                    realline.append(val)
                    realindex +=1
                    continue
                self.cols[realindex].obj + self.convert(val)
                realline.append(val)
                realindex +=1
            else:
                realindex+=1
            index += 1
        self.rows.append(line)
        self.count += 1

    def delete_row(self, line):
        """
        removes the effects of a row from a Sample
        """
        if self.fileline < 1:
            print("Cant delete empty sample")
            return
        self.fileline -= 1
        if len(line) < self.linesize:
            print("Line", self.fileline, "has an error")
            return
        realline = []
        realindex = 0
        index = 0
        for val in line:
            if index + 1 not in self.skip:
                if val == "?":
                    continue
                self.cols[realindex].obj - self.convert(val)
                realindex +=1
            else:
                realindex+=1
            index += 1
        self.rows.append(line)
        self.count += 1

    def __add__(self, line):
        """
        add method for Sample
        """
        if len(self.header) > 0:
            self.insert_row(line)
        else:
            self.create_cols(line)

    def __sub__(self, line):
        """
        sub method for Sample
        """
        self.delete_row(line)

    def s_goals(self, rows = None):
        if rows is None:
            rows = self.rows
        if len(self.y) == 0:
            return []
        if len(rows) == 0:
            return []
        return [ self.s_median(c, rows) for c in self.y ]

    def s_median(self, col, rows = None):
        if rows is None:
            rows = self.rows
        data = [ self.convert(r[col-1]) for r in rows ]
        return median(data)

    def dump(self, filename = "tabledump.txt"):
        """
        dumps Sample data into a file
        """
        f = open(filename, 'w')
        f.write("Dump table:"+"\n")
        f.write("t.cols"+"\n")
        for i, col in enumerate(self.cols):
            if i+1 in self.skip:
                continue
            if col.isnum:
                f.write("|  "+str(col.oid)+"\n")
                f.write("|  |  add: Num"+str(i+1)+"\n")
                f.write("|  |  col: "+str(col.oid)+"\n")
                f.write("|  |  hi: "+str(col.obj.hi)+"\n")
                f.write("|  |  lo: "+str(col.obj.lo)+"\n")
                f.write("|  |  m2: "+str(col.obj.m2)+"\n")
                f.write("|  |  mu: "+str(col.obj.mu)+"\n")
                f.write("|  |  n: "+str(col.obj.n)+"\n")
                f.write("|  |  oid: "+str(col.oid)+"\n")
                f.write("|  |  sd: "+str(col.obj.sd)+"\n")
                f.write("|  |  txt: "+str(col.txt)+"\n")
            else:
                f.write("|  " + str(col.oid) + "\n")
                f.write("|  |  add: Sym"+ str(i+1) + "\n")
                for k, v in col.obj.cnt.items():
                    f.write("|  |  |  " + str(k) + ": " + str(v) + "\n")
                f.write("|  |  col: "+str(col.oid)+"\n")
                f.write("|  |  mode: "+str(col.obj.mode)+"\n")
                f.write("|  |  most: "+str(col.obj.most)+"\n")
                f.write("|  |  n: " + str(col.obj.n) + "\n")
                f.write("|  |  oid: " + str(col.oid) + "\n")
                f.write("|  |  txt: " + str(col.txt) + "\n")

        f.write("t.my: "+"\n")
        f.write("|  len(cols): " + str(len(self.cols))+"\n")
        f.write("|  y" + "\n")
        for v in self.y:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        if self.klass != -1:
            f.write("|  klass" + "\n")
            f.write("|  |  " + str(self.klass) + "\n")
        f.write("|  nums" + "\n")
        for v in self.nums:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  syms" + "\n")
        for v in self.syms:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  w" + "\n")
        for k, v in self.w.items():
            if v not in self.skip:
                f.write("|  |  " + str(k) + ": "+str(v)+"\n")
        f.write("|  x" + "\n")
        for v in self.x:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  xnums" + "\n")
        for v in self.xnums:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  xsyms" + "\n")
        for v in self.xsyms:
            if v not in self.skip:
                f.write("|  |  " + str(v) + "\n")
        f.close()

    def dumpSorted(self, filename = "tabledump.txt"):
        """
        dumps Sample data sorted
        """
        f = open(filename, 'w')
        f.write(str(self.header)+"\n")
        self.sorted()
        for i, row in enumerate(self.rows):
            if i < 5:
                f.write(str(row) + "\n")
            if i == 5:
                f.write("\n")
            if i >= (len(self.rows) - 6):
                f.write(str(row) + "\n")

    def clone(self):
        """
        returns a clone of a Sample
        """
        new = Sample(self.oid)
        new.create_cols(self.header)
        new.x = self.x
        new.y = self.y
        new.xnums = self.xnums
        new.xsyms = self.xsyms
        new.klass = self.klass
        new.skip = self.skip
        new.w = self.w
        new.fileline = self.fileline
        new.linesize = self.linesize
        new.header = self.header 
        return new

    def dist(self, r1, r2):
        d,n = 0,1E-32
        cols = self.x
        for col in cols:
            n += 1
            a = self.convert(r1[col-1])
            b = self.convert(r2[col-1])
            d += self.cols[col-1].dist(a, b)**2
        return (d/n)**(1/2)

    def sortGroups(self, groups):
        repr = [ [0 for c in self.cols] for g in groups ]
        # Set the correct median values for the column
        for i, g in enumerate(groups):
            for c in self.y:
                repr[i][c - 1] = g.s_median(c)
        new_list = [[i,x] for i,x in enumerate(repr)]
        self.idxQuickSort(new_list, 0, len(new_list)-1)
        ordered_groups = [groups[i] for i, _ in new_list]
        # Fourth, if verbose, we show the results
        self._print_groups(ordered_groups)

        return ordered_groups

    def neighbors(self, r1, rows):
        a = []
        for r2 in rows:
            if r1 != r2:
                a.append( (self.dist(r1, r2), r2) )
        
        def sort_fun(x, y):
            if x[0] == y[0]: return 0
            elif x[0] < y[0]: return -1
            else: return 1

        a.sort(key = cmp_to_key(sort_fun), reverse = False)
        return a

    def shuffle(self, rows, samples):

        samples = min(len(rows), samples)
        idx = sample( [i for i in range(len(rows))], k = samples )
        return [ rows[i] for i in idx ]

    def faraway(self, row, rows = None):
            rows = self.rows if rows is None else rows 
            samples = 128
            all = self.neighbors(row, rows = self.shuffle( rows, samples = samples ))
            return all[-1][1]

    def div1(self, rows):

        zero = self.shuffle(rows, samples = 1)[0]
        A = self.faraway(zero, rows)
        B = self.faraway(A, rows)
        c = self.dist(A, B) 

        projection = []
        for C in rows:
            a = self.dist(C, B) 
            b = self.dist(A, C) 
            proj = (b**2 + c**2 - a**2) / (2*c)
            projection.append(proj)
        new_list = [[i,x] for i,x in enumerate(projection)]
        new_list.sort(key=lambda x: x[1])
        sorted = [rows[i] for i, _ in new_list]
        mid = len(sorted) // 2
        return sorted[:mid], sorted[mid:]

    def divs(self):
        return self._divs( self.rows, 1, math.ceil( self.count**(1/2) ))
    
    def _divs(self, rows, level, min_leaf_size):
        if len(rows) < min_leaf_size:
            new = self.clone()
            for row in rows:
                new + row
            self._print_leaf(rows, level)
            return [new]
        self._print_node(rows, level)
        east, west = self.div1(rows)
        east = self._divs(east, level + 1, min_leaf_size)
        west = self._divs(west, level + 1, min_leaf_size)
        return east + west

    def _print_node(self, rows, level):
        text = "|.. " * level
        text += f"n={len(rows)} c={self.calc_c(rows) : .2f}"
        print(text)
    
    def _print_leaf(self, rows, level):
        text = "|.. " * level
        text += f"n={len(rows)} c={self.calc_c(rows) : .2f}"
        text += " " * 5
        text += "goals = ["
        if self.klass != -1:
            text += "-"
        elif len(self.y) > 0:
            data = self.s_goals(rows)
            data = [ f"{d : .1f}" for d in data ]
            text += ",".join(data)
        else:
            text += "-"
        text += "]"
        print(text)

    def _print_groups(self, groups):
        if (len(groups) > 1):
            names = [ self.cols[y-1].txt for y in self.y ]
            space = [len(n) + 5 for n in names]
            s = ""

            for n, sp in zip(names, space):
                s += f"{n : <{sp}}"
            s += "\n"

            for i, g in enumerate(groups):
                median = g.s_goals()
                for m, sp in zip(median, space):
                    s += f"{m : <{sp}.1f}"
                if i == 0:
                    s += " <== best"
                if i == len(groups) - 1:
                    s += " <== worst"
                s += "\n"
            
            print(s)

    def discretize(self):
        groups = self.divs()
        groups = self.sortGroups(groups)

        feature_ranges = []

        best, worst = groups[0], groups[-1]
        for good, bad in zip(best.x, worst.x):
            range = []
            for res in best.cols[good-1].discretize(worst.cols[bad-1]):
                range += [res]
            if len(range) > 0:
                feature_ranges += [range]
        
        self._show_discretized_ranges(feature_ranges, best, worst)

        return feature_ranges
        
    def _show_discretized_ranges(self, feature_ranges, best, worst):
        for rang in feature_ranges:
            for r in rang:
                print(r)
            print("")

        # Show some values from best and worst
        print("Best")
        for i in range(min(6, len(best.rows))):
            print("best", best.rows[i])
        print()
        for i in range(min(6, len(worst.rows))):
            print("worst", worst.rows[i])

    def calc_c(self, rows = None):
        rows = self.rows if rows is None else rows
        zero = self.shuffle(rows, 1)[0]
        one = self.faraway(zero, rows)
        two = self.faraway(one, rows)
        return self.dist(one, two)

    def norm_c(self, rows):
        return self.calc_c(rows) / self.calc_c(self.rows)


def main():
    lines = Sample.read("sin21/data/auto93.csv")
    sample = Sample(0)
    ls = sample.linemaker(lines)
    for l in ls:
        sample + l
    g = sample.divs()
    # sample.dump()
    sample.discretize()


    


if __name__ == '__main__':
    main()