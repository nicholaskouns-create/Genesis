from fractions import Fraction as F
from itertools import product
from collections import Counter
w=Counter(sum(t) for t in product(range(-2,3),repeat=3))
mult={J:w[J]-w[J+1] for J in range(7)}
m=[mult[J] for J in range(7)]; lam=[J*(J+1) for J in range(7)]; dim=[2*J+1 for J in range(7)]
N=sum(a*b for a,b in zip(m,dim)); mu=F(sum(a*b*l for a,b,l in zip(m,dim,lam)),N); var=F(sum(a*b*l*l for a,b,l in zip(m,dim,lam)),N)-mu*mu
keep=[l for l in lam if (l-mu)**2==var]; D=sum(a*b for a,b,l in zip(m,dim,lam) if l in keep)
gap=min(abs((l-mu)**2-var)/var for l in lam if l not in keep)
print("python-fractions",[a*b for a,b in zip(m,dim)],N,mu,var,keep,D,gap)
