# Independent check of the decomposition D_i = E_i + J_i (Lemma 2) and of J_i <= (2/ln 2) Inf_i (Lemma 3).
import numpy as np, sys, math
def wht(F):
    F=F.astype(float).copy(); n=F.shape[-1].bit_length()-1; h=1
    while h<F.shape[-1]:
        F=F.reshape(F.shape[0],-1,2,h); a=F[:,:,0,:].copy(); b=F[:,:,1,:].copy()
        F[:,:,0,:]=a+b; F[:,:,1,:]=a-b; F=F.reshape(F.shape[0],-1); h*=2
    return F/F.shape[-1]
def H(p):
    p=np.where(p>1e-300,p,1.0); return -(p*np.log2(p)).sum(-1)
def run(T,n):  # T: (m, 2^n) of +-1
    c=wht(T); mu=c**2; Hf=H(mu); N=1<<n; idx=np.arange(N)
    worstJ=0; worstJarg=None; bestcoord=[]; maxid=0
    for i in range(n):
        lo=idx[(idx>>i)&1==0]; hi=lo|(1<<i)
        a=c[:,lo]; b=c[:,hi]; inf=(b**2).sum(1)
        mp=(a+b)**2; mm=(a-b)**2; m=(a*a+b*b)
        E=(m*np.where(m>0,0,0)).sum(1)  # placeholder
        q=np.where(m>0,b*b/np.where(m>0,m,1),0)
        hq=np.where((q>0)&(q<1),-(q*np.log2(np.where(q>0,q,1))+(1-q)*np.log2(np.where(q<1,1-q,1))),0)
        E=(m*hq).sum(1)
        J=H(m)-0.5*H(mp)-0.5*H(mm)
        # D_i by actual restriction of truth tables: coordinate i of x is bit i of index
        fp=T[:,hi]; fm=T[:,lo]
        D=Hf-0.5*H(wht(fp)**2)-0.5*H(wht(fm)**2)
        maxid=max(maxid,np.abs(D-E-J).max())
        ok=inf>1e-12
        r=np.where(ok,J/np.where(ok,inf,1),0)
        k=r.argmax()
        if r[k]>worstJ: worstJ=r[k]; worstJarg=(i,int(k))
        bestcoord.append(np.where(ok,(E)/np.where(ok,inf,1),np.inf))
    best=np.min(np.array(bestcoord),0); nonconst=np.isfinite(best)
    return maxid,worstJ,worstJarg,best[nonconst].max()
n=4; N=16; m=1<<N
T=1-2*((np.arange(m)[:,None]>>np.arange(N)[None,:])&1)
print("n=4 all:", run(T,n), " 2/ln2=",2/math.log(2))
rng=np.random.default_rng(1)
for n in (5,6,7):
    for p in (0.5,0.2,0.05):
        T=np.where(rng.random((40000,1<<n))<p,-1,1)
        r=run(T,n); print(f"n={n} p={p}: max|D-E-J|={r[0]:.2e} max J/Inf={r[1]:.4f} max_f min_i E_i/Inf_i={r[3]:.4f}")
