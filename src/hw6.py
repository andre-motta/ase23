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
import numpy as np

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
    mean = sum(items) / n
    deviations = [(x - mean) ** 2 for x in items]
    return sum(deviations) / n

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
        current = [g[1] for g in groups[i]]
        next = [g[1] for g in groups[i + 1]]
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

class FFTNode:
    def __init__(self,full_policy,surviving_data,ranges):
        if(full_policy == None):
            self.txt = "Last leaf"
            self.rows = surviving_data
            self.support = len(surviving_data)
            self.range = None
            self.policy = None
            self.next = None
        else:
            pass_data = []
            self.policy = full_policy[0]
            self.rows = []
            self.support = 0
            if self.policy:
                self.range = ranges[0]
                if len(ranges) > 1:
                    ranges = ranges[1:]
            else:
                self.range = ranges[-1]
                if len(ranges) > 1:
                    ranges = ranges[:-1]
            self.txt = self.range['txt']
            if self.txt[0].isupper():
                if self.range['first']:
                    for row in surviving_data:
                        if row[self.range['at']] != '?' and Sample.compiler(row[self.range['at']])(row[self.range['at']]) <= self.range['hi']:
                            self.rows.append(row)
                            self.support += 1
                        else:
                            pass_data.append(row)
                elif self.range['last']:
                    for row in surviving_data:
                        if  row[self.range['at']] != '?' and Sample.compiler(row[self.range['at']])(row[self.range['at']]) > self.range['lo']:
                            self.rows.append(row)
                            self.support += 1
                        else:
                            pass_data.append(row)
                else:
                    for row in surviving_data:
                        if  row[self.range['at']] != '?' and self.range['lo'] <= Sample.compiler(row[self.range['at']])(row[self.range['at']]) < self.range['hi']:
                            self.rows.append(row)
                            self.support += 1
                        else:
                            pass_data.append(row)
            else:
                for row in surviving_data:
                    if row[self.range['at']] != '?' and Sample.compiler(row[self.range['at']])(row[self.range['at']]) == self.range['lo']:
                        self.rows.append(row)
                        self.support += 1
                    else:
                        pass_data.append(row)

            if len(full_policy) > 1:
                full_policy = full_policy[1:]
                self.next = FFTNode(full_policy, pass_data, ranges)
            else:
                self.next = FFTNode(None,pass_data,None)

           
class FFT():
    def __init__(self, all, conf, branch,branches,stop = None, level=0):

        self.my = conf
        #stop = stop or 2*len(all.rows)**self.my.bins
        stop = stop or 2*len(all.rows)**0.5
        
        bins = all.discretize()

        bestIdea = None
        worstIdea = None
        bestValues = self.values("plan", bins)
        worstValues = self.values("monitor", bins)

        if len(bestValues)>0 and len(worstValues)>0:
            bestIdea   = bestValues[-1][1]
            worstIdea  = worstValues[-1][1]
            
            for yes,no,idea in [(1,0,bestIdea), (0,1,worstIdea)]:
                leaf,tree = all.clone(), all.clone()
                for row in all.rows:
                    if self.match(idea, row):
                        leaf + row
                    else:
                        tree + row
                b1 = copy.deepcopy(branch)
                b1 += [Bag(at=idea.at -1 , lo=idea.lo, hi=idea.hi,
                                type=yes, txt="if "+self.show(idea)+" then with support = "+ str(len(leaf.rows)) , 
                                then=leaf.ys(), n=len(leaf.rows))]
                if len(tree.rows) <= stop or level > random.randrange(5,9,1):
                    b1  += [Bag(type=no, txt="exit node support = "+ str(len(tree.rows)), then=tree.ys(), n= len(tree.rows))]
                    branches += [b1]
                else:
                    FFT(tree,conf,b1,branches,stop=stop,level=level+1)
            
    
    def match(self, bin, row):
        v=row[bin.at]             
        if   v=="?"   : return True      
        elif bin.first: return Sample.compiler(v)(v) <= bin.hi
        elif bin.last : return Sample.compiler(v)(v) >= bin.lo
        else          : return bin.lo <= Sample.compiler(v)(v) <= bin.hi

    def show(self,bin):
        if   bin.lo == bin.hi: return f"{bin.txt} == {bin.lo}"  
        elif bin.first: return f"{bin.txt} <= {bin.hi}"
        elif bin.last : return f"{bin.txt} >= {bin.lo}"
        else          : return f"{bin.lo} <= {bin.txt} <= {bin.hi}"

    def value(self, rule, bin):
        # s = self.my.support
        s = 2
        rules = Bag(plan    = lambda b,r: b**s/(b+r) if b>r else 0,  
                monitor = lambda b,r: r**s/(b+r) if r>b else 0,  
                novel   = lambda b,r: 1/(b+r))                   

        if bin.rests == 0 or bin.bests == 0:
            return rules[rule](bin.best, bin.rest)
        return rules[rule](bin.best/bin.bests, bin.rest/bin.rests)
  
    def values(self,rule,bins):
        bins = [(self.value(rule,bin), bin) for bin in bins]
        tmp = [(n,bin) for n,bin in bins if n > 0]
        return sorted(tmp, key=lambda tuple: tuple[0]) 

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
                    bests=self.n,
                    rest = other.get(val),
                    rests = other.n,
                    first = False, last = False )
    def mid(self):
        return self.mode

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
                        bests = self.n, 
                        rest = sum(1 for x in counts if x == 0),
                        rests = other.n,
                        first = (n == 0), last = (n == len(ranges)))
    def mid(self):
        return round(self.mu, 1)

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
    def mid(self):
        return self.obj.mid()

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
        # self._print_groups(ordered_groups)

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
            # self._print_leaf(rows, level)
            return [new]
        # self._print_node(rows, level)
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
                feature_ranges += range
        
        # self._show_discretized_ranges(feature_ranges, best, worst)

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

    def ys(self):
        return [self.cols[goal-1].mid() for goal in self.y]


def main():
    lines = Sample.read("data/auto93.csv")
    sample = Sample(0)
    ls = sample.linemaker(lines)
    for l in ls:
        sample + l
    branches = []
    branch1 = []
    # ranges = sample.discretize()
    # ranked_ranges = sorted(sorted(ranges,key=lambda x: x['rest']), key=lambda x: x['best'], reverse=True)
    # fft = FFTNode([0,0,0,1], sample.rows, ranked_ranges)
    fft = FFT(sample, None, branch1, branches)
    for i, branch in enumerate(branches):
        print("FFT", i)
        for b in branch:
            print(b.txt)
        print()
    a = 2
        



if __name__ == '__main__':
    main()