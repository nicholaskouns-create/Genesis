from fractions import Fraction as F
from math import comb, isqrt, factorial
from sympy.utilities.iterables import partitions
def sel(spec):  # spec: list of (eigenvalue, multiplicity) exact
    N=sum(m for _,m in spec); mu=F(sum(l*m for l,m in spec),N); var=F(sum(l*l*m for l,m in spec),N)-mu*mu
    kept=[(l,m) for l,m in spec if (l-mu)**2==var]
    lo=any(l<mu for l,_ in kept); hi=any(l>mu for l,_ in kept)
    return N,mu,var,sum(m for _,m in kept),lo and hi,lo or hi
def merge(s):
    d={}
    for l,m in s: d[l]=d.get(l,0)+m
    return sorted(d.items())
out=[]
for n in range(2,40):
    s=merge([(n-2*k,comb(n,k)) for k in range(n+1)]); r=sel(s)
    if r[4]: out.append((f'hypercube Q_{n} adjacency',r))
for q in range(3,8):
    for n in range(1,25):
        s=merge([(n*(q-1)-q*k,comb(n,k)*(q-1)**k) for k in range(n+1)]); r=sel(s)
        if r[4]: out.append((f'Hamming H({n},{q}) adjacency',r))
for n in range(4,40):
    for k in range(2,n//2+1):
        s=merge([((k-j)*(n-k-j)-j,comb(n,j)-(comb(n,j-1) if j else 0)) for j in range(k+1)]); r=sel(s)
        if r[4]: out.append((f'Johnson J({n},{k}) adjacency',r))
def hooks(p):
    c=[sum(1 for x in p if x>j) for j in range(p[0])]; h=1
    for i,x in enumerate(p):
        for j in range(x): h*=x-j+c[j]-i-1
    return h
for n in range(2,13):
    s=[]
    for part in partitions(n):
        p=sorted([k for k,v in part.items() for _ in range(v)],reverse=True)
        f=factorial(n)//hooks(p); content=sum(j-i for i,x in enumerate(p) for j in range(x)); s.append((content,f*f))
    r=sel(merge(s))
    if r[4]: out.append((f'S_{n} transposition class sum, regular rep',r))
# SU(2) V_j^{(x)n} Casimir, integer and half-integer j
def su2(twoj,n):
    w={0:1}
    for _ in range(n):
        nw={}
        for a,c in w.items():
            for m in range(-twoj,twoj+1,2): nw[a+m]=nw.get(a+m,0)+c
        w=nw
    top=twoj*n; s=[]
    for tJ in range(top%2,top+1,2):
        m=w.get(tJ,0)-w.get(tJ+2,0)
        if m: s.append((F(tJ*(tJ+2),4),m*(tJ+1)))
    return s
for n in range(2,6):
    for twoj in range(1,(26 if n==3 else 14)):
        r=sel(su2(twoj,n))
        if r[4]: out.append((f'SU(2) spin {F(twoj,2)} tensor power {n}',r))
print(f'{"carrier":44s} {"dim":>12s} {"mu":>6s} {"sigma^2":>8s} {"kept":>10s}  fraction')
for name,(N,mu,var,k,_,_) in out: print(f'{name:44s} {N:12d} {str(mu):>6s} {str(var):>8s} {k:10d}  {k/N:.4f}')
