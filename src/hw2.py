import math
import re
from collections import defaultdict
import random
from operator import itemgetter
r = random.random
seed = random.seed

"""
This code reads allows users to read in a csv file, 
and retain information on classes, weights, numbers, 
symbols and goals.

"""


class Sym:
    """
    Sym class represents the symbolic information
    """
    n=0
    most = 0
    mode = ""

    def __init__(self, data=None):
        """
        Construct a new Sym object
        """
        self.cnt = defaultdict(int)
        if data != None:
            for val in data:
                self + val

    def __add__(self, v):
        """
        Add method for Sym
        """
        self.n += 1
        self.cnt[v] += 1
        tmp = self.cnt[v]
        if tmp > self.most:
            self.most = tmp
            self.mode = v
        return v

    def __sub__(self, x):
        old = self.cnt.get(x, 0)
        if old > 0:
            self.cnt[x] = old - 1

    def variety(self):
        return self.syment()

    def xpect(self, j):
        n = self.n + j.n
        return self.n / n * self.variety() + j.n / n * j.variety()

    def syment(self):
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

    @staticmethod
    def symlike(self,x,prior,m):
        f = self.cnt[x]
        return (f + m*prior)/(self.n + m)

# Sym class represents the numeric information
class Num:
    n = 0
    mu = 0
    m2 = 0
    sd = 0
    lo = float('inf')
    hi = -float('inf')

    def __init__(self, data=None):
        if data != None:
            for val in data:
                self + val
        
    def variety(self):
        return self.sd

    def xpect(self, j):
        n = self.n + j.n
        return self.n / n * self.variety() + j.n / n * j.variety()

    def _numSd(self):
        if self.m2 < 0:
            return 0
        if self.n < 2:
            return 0
        return math.sqrt(self.m2/(self.n - 1))

    def numNorm(self, x):
        return (x - self.lo)/(self.hi - self.lo + 10e-32)

    def __sub__(self, v):
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
        self.n += 1
        if v < self.lo:
            self.lo = v
        if v > self.hi:
            self.hi = v
        d = v - self.mu
        self.mu += d / self.n
        self.m2 += d * (v - self.mu)
        self.sd = self._numSd()
        return v

    @staticmethod
    def numlike(self, x):
        var = self.sd**2
        denom = math.sqrt(math.pi * 2 * var)
        num = (2.71828**(-(x-self.mu)**2)/(2*var+0.0001))
        return num/(denom+10**(-64)) + 10**(-64)

# Col class keeps information on the columns
class Col:
    def __init__(self, oid, txt, obj, isnum):
        self.oid, self.txt, self.obj, self.isnum = oid, txt, obj, isnum

    def __add__(self, v):
        self.obj + v

    def __sub__(self, v):
        self.obj - v

    def variety(self):
        return self.obj.variety()

# Sample class keeps a bundle of columns
class Sample:
    def __init__(self, oid):
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
        lines = []
        for line in s.splitlines():
            lines.append(line)
        return lines

    @staticmethod
    def compiler(x):
        try:
            int(x)
            return int
        except:
            try:
                float(x)
                return float
            except ValueError:
                return str

    def convert(self, x):
        f = self.compiler(x)
        return f(x)

    def create_cols(self, line):
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
                self.cols.append(Col(index+1, str(''.join(c for c in val if not c in ['?'])), Num(), True))
            else:
                self.syms.append(index + 1)
                cols.append(''.join(c for c in val if not c in ['?']))
                self.cols.append(Col(index+1, str(''.join(c for c in val if not c in ['?'])), Sym(), False))

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
                    continue
                self.cols[realindex].obj + self.convert(val)
                realline.append(val)
                realindex +=1
            else:
                realindex+=1
            index += 1
        self.rows.append(line)
        self.count += 1

    def __add__(self, line):
        if len(self.header) > 0:
            self.insert_row(line)
        else:
            self.create_cols(line)

    def dump(self, filename = "tabledump.txt"):
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

# Experiment ran in 00.31s for pom3a.txt
def main():
    lines = Sample.read("data/pom3a.txt")
    sample = Sample(0)
    ls = sample.linemaker(lines)
    for l in ls:
        sample + l

    
    sample.dump("outpart3.txt")


if __name__ == '__main__':
    main()