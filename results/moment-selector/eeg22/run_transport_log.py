import sys, json, numpy as np
sys.path.insert(0,'/home/claude/repos/Genesis')
from scipy.signal import butter, sosfiltfilt, resample
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import OAS
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from genesis.fractal import FractalIntentSignature
from genesis.coherence import CoherenceProjection
import mne; mne.set_log_level('ERROR')
D=sys.argv[1]; TAUS=(0.5,); NR=20
lda=lambda: LDA(solver='lsqr',shrinkage='auto')

def load(s):
    z=np.load(f'{D}/A0{s}T.npz',allow_pickle=False)
    x=np.nan_to_num(z['s'][:,:22].astype(float)); et=z['etyp'].ravel(); ep=z['epos'].ravel()
    x=sosfiltfilt(butter(4,[8,32],btype='band',fs=250,output='sos'),x,axis=0)
    X=[];y=[]
    for t,p in zip(et,ep):
        if t in (769,770) and p+1000<=len(x):
            X.append(resample(x[p:p+1000].T,512,axis=1)); y.append(int(t==770))
    return np.array(X),np.array(y)

def moment_select(C,tau):
    lam,V=np.linalg.eigh(C); lam=np.log(np.maximum(lam,1e-12)); d=len(lam)
    mu=lam.mean(); sig=np.sqrt(max((lam**2).mean()-mu**2,0))
    K=(lam-mu)**2-sig**2
    keep=np.where(np.abs(K)<=tau*sig**2)[0]; fb=False
    if len(keep)==0:
        keep=np.unique([np.argmin(abs(lam-(mu-sig))),np.argmin(abs(lam-(mu+sig)))]); fb=True
    return V[:,keep],dict(k=len(keep),d=d,mu=mu,sig=sig,fb=fb,lo=int((lam[keep]<mu).sum()),hi=int((lam[keep]>mu).sum()))

def auc(Ftr,ytr,Fte,yte):
    m=lda().fit(Ftr,ytr); return roc_auc_score(yte,m.decision_function(Fte))
def logvar(X,W): return np.log(np.einsum('ncT,ck->nkT',X,W).var(axis=2)+1e-30)

res={}
for s in range(1,10):
    X,y=load(s); F=np.nan_to_num(FractalIntentSignature(sfreq=128.0,n_jobs=4).transform(X))
    R={}; add=lambda k,v: R.setdefault(k,[]).append(v)
    rng=np.random.default_rng(47)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=47).split(F,y):
        # Carrier A
        sc=StandardScaler().fit(F[tr]); A,B=sc.transform(F[tr]),sc.transform(F[te]); d=A.shape[1]
        C=OAS().fit(A).covariance_
        add('A none',auc(A,y[tr],B,y[te]))
        cp=CoherenceProjection().fit(A,y[tr]); add('A fisher47',auc(cp.transform(A),y[tr],cp.transform(B),y[te]))
        lamv,Vv=np.linalg.eigh(C)
        for tau in TAUS:
            Q,info=moment_select(C,tau); add(f'A moment {tau}',auc(A@Q,y[tr],B@Q,y[te])); add(f'A frac {tau}',info['k']/d); add(f'A fb {tau}',info['fb'])
            if tau==0.5:
                k=info['k']; add('A lo',info['lo']); add('A hi',info['hi']); add('A sig',info['sig'])
                rr=[]
                for _ in range(NR):
                    G=np.linalg.qr(rng.standard_normal((d,k)))[0]; rr.append(auc(A@G,y[tr],B@G,y[te]))
                add('A random',np.mean(rr)); T=Vv[:,-k:]; add('A topk',auc(A@T,y[tr],B@T,y[te]))
        # Carrier B
        covs=np.array([x@x.T/np.trace(x@x.T) for x in X[tr]]); Cb=covs.mean(0); lamb,Vb=np.linalg.eigh(Cb)
        for tau in TAUS:
            W,info=moment_select(Cb,tau); add(f'B moment {tau}',auc(logvar(X[tr],W),y[tr],logvar(X[te],W),y[te])); add(f'B frac {tau}',info['k']/22); add(f'B fb {tau}',info['fb'])
            if tau==0.5:
                k=info['k']; add('B lo',info['lo']); add('B hi',info['hi']); rr=[]
                for _ in range(NR):
                    G=np.linalg.qr(rng.standard_normal((22,k)))[0]; rr.append(auc(logvar(X[tr],G),y[tr],logvar(X[te],G),y[te]))
                add('B random',np.mean(rr)); T=Vb[:,-k:]; add('B topk',auc(logvar(X[tr],T),y[tr],logvar(X[te],T),y[te]))
        add('B all22',auc(logvar(X[tr],np.eye(22)),y[tr],logvar(X[te],np.eye(22)),y[te]))
        csp=CSP(n_components=8).fit(X[tr],y[tr]); add('B csp',auc(csp.transform(X[tr]),y[tr],csp.transform(X[te]),y[te]))
        cv=Covariances('oas'); ts=TangentSpace(metric='riemann'); Ttr=ts.fit_transform(cv.transform(X[tr])); Tte=ts.transform(cv.transform(X[te]))
        lr=LogisticRegression(C=1.0,max_iter=1000).fit(Ttr,y[tr]); add('B tangent',roc_auc_score(y[te],lr.decision_function(Tte)))
    res[s]={k:float(np.mean(v)) for k,v in R.items()}; res[s]['n']=int(len(y)); res[s]['d']=int(F.shape[1])
    print(s,{k:round(v,3) for k,v in res[s].items()},flush=True)
json.dump(res,open('results_log.json','w'),indent=1)
def col(k): return np.array([res[s][k] for s in res])
print('\nMEANS'); 
for k in res[1]: print(f'  {k:16s} {col(k).mean():.3f} ± {col(k).std(ddof=1):.3f}')
def test(a,b,name):
    dlt=col(a)-col(b); p=wilcoxon(dlt,alternative='greater').pvalue if np.any(dlt!=0) else 1.0
    print(f'{name}: Δ={dlt.mean():+.3f}  wins {int((dlt>0).sum())}/9  p={p:.4f}  {"SUPPORTED" if p<0.0125 else "not supported"}')
print('\nTESTS (alpha 0.0125)')
test('A moment 0.5','A none','M1 A moment > no projection')
test('A moment 0.5','A random','M2 A moment > random same rank')
test('B moment 0.5','B random','M3 B moment > random same rank')
test('B moment 0.5','B csp','M4 B moment > CSP')
