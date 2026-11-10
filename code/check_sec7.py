# Independent checks of Lemma 3 (sharp constant) and Proposition 7 (bent point replacement).
import numpy as np, math
th=np.linspace(1e-6,np.pi/4-1e-6,2000001); t=np.sin(2*th)
jsd=(t*np.arctanh(t)+0.5*np.log1p(-t*t))/math.log(2)   # 1 - h((1+t)/2) in bits, accurate near t = 0
r=jsd/np.sin(th)**2; k=r.argmax(); print("sup ratio %.6f at theta %.2e ; 2/ln2=%.6f"%(r[k],th[k],2/math.log(2)))
def wht(v):
    v=v.astype(float).copy(); h=1
    while h<len(v):
        v=v.reshape(-1,2,h); a=v[:,0].copy(); b=v[:,1].copy(); v[:,0]=a+b; v[:,1]=a-b; v=v.reshape(-1); h*=2
    return v/len(v)
def E_and_inf(f,n,i):
    c=wht(f); idx=np.arange(1<<n); lo=idx[(idx>>i)&1==0]; a=c[lo]; b=c[lo|(1<<i)]; m=a*a+b*b
    q=np.where(m>0,b*b/np.where(m>0,m,1),0); qq=np.clip(q,1e-300,1); q1=np.clip(1-q,1e-300,1)
    hq=np.where((q>0)&(q<1),-(q*np.log2(qq)+(1-q)*np.log2(q1)),0); return (m*hq).sum(),(b*b).sum()
for k in (2,4,6,8,10):   # bent inner product on k vars (y), x_i = bit k; f = h(y) except at y*=0 where f = x_i
    n=k+1; N=1<<n; x=np.arange(N); y=x&((1<<k)-1); xi=(x>>k)&1
    hy=np.array([(-1)**bin((v&((1<<(k//2))-1)) & (v>>(k//2))).count('1') for v in range(1<<k)])
    f=hy[y].copy(); f[y==0]=1-2*xi[y==0]
    E,inf=E_and_inf(f,n,k); print(f"k={k}: E_i/Inf_i at the replaced coordinate = {E/inf:.3f}")
