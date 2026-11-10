import sympy as sp
x=sp.symbols('x'); ch=lambda j: sum(x**k for k in range(-2*j,2*j+1,2))
p=sp.expand(ch(2)**3); c=sp.Poly(sp.expand(p*x**12),x).all_coeffs()[::-1]; w={k-12:c[k] for k in range(25)}
m=[w[2*J]-w.get(2*J+2,0) for J in range(7)]; lam=[J*(J+1) for J in range(7)]; d=[m[J]*(2*J+1) for J in range(7)]
mu=sp.Rational(sum(a*l for a,l in zip(d,lam)),sum(d)); var=sp.Rational(sum(a*l*l for a,l in zip(d,lam)),sum(d))-mu**2
K=sp.factor((x-mu)**2-var); print("sympy",d,mu,var,K,sum(a for a,l in zip(d,lam) if K.subs(x,l)==0))
