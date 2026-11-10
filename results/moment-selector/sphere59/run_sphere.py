import sys,json,itertools,warnings,numpy as np,scipy.io as sio
warnings.filterwarnings('ignore')
from scipy.special import sph_harm_y
from scipy.signal import butter,sosfiltfilt
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import mne; mne.set_log_level('ERROR')
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
D=sys.argv[1]; L=6; SETS=[(a,b) for a in range(L+1) for b in range(a,L+1)]
lda=lambda: LDA(solver='lsqr',shrinkage='auto')
def geometry(nfo):
    clab=[str(c[0]) for c in nfo['clab'].ravel()]; x=nfo['xpos'].ravel().astype(float); y=nfo['ypos'].ravel().astype(float)
    r=np.hypot(x,y); phi=np.arctan2(y,x)
    pos=mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    cz=pos['Cz']/np.linalg.norm(pos['Cz']); idx=[i for i,c in enumerate(clab) if c in pos and r[i]>0]
    th=np.array([np.arccos(np.clip(pos[clab[i]]@cz/np.linalg.norm(pos[clab[i]]),-1,1)) for i in idx])
    k=(r[idx]@th)/(r[idx]@r[idx]); res=np.sqrt(np.mean((k*r[idx]-th)**2))
    return k*r,phi,k,res,len(idx)
def realY(theta,phi):
    cols=[];deg=[]
    for l in range(L+1):
        for m in range(-l,l+1):
            Y=sph_harm_y(l,abs(m),theta,phi)
            cols.append(Y.real if m==0 else (np.sqrt(2)*Y.real if m>0 else np.sqrt(2)*Y.imag)); deg.append(l)
    return np.array(cols).T,np.array(deg)
def auc(Ftr,ytr,Fte,yte):
    m=lda().fit(Ftr,ytr); return roc_auc_score(yte,m.decision_function(Fte))
out={}
for s in 'abfgcde':
    m=sio.loadmat(f'{D}/BCICIV_calib_ds1{s}.mat'); nfo=m['nfo'][0,0]; mrk=m['mrk'][0,0]
    theta,phi,k,res,nfit=geometry(nfo); Y,deg=realY(theta,phi)
    A=np.linalg.solve(Y.T@Y+0.01*np.trace(Y.T@Y)/49*np.eye(49),Y.T)      # 49 x 59
    x=m['cnt'].astype(float); x=x-x.mean(1,keepdims=True)
    x=sosfiltfilt(butter(4,[8,32],btype='band',fs=100,output='sos'),x,axis=0)
    pos=mrk['pos'].ravel(); y=(mrk['y'].ravel()>0).astype(int)
    X=np.array([x[p:p+400].T for p in pos])                               # trials x 59 x 400
    Cf=np.einsum('kc,nct->nkt',A,X); V=Cf.var(axis=2); F=np.log(V+1e-30)  # trials x 49
    Fch=np.log(X.var(axis=2)+1e-30)
    R={}; add=lambda key,v: R.setdefault(key,[]).append(v); sel=[]
    for seed in range(47,52):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(F,y):
            p=np.array([V[tr][:,deg==l].sum() for l in range(L+1)]); p=p/p.sum(); lam=np.array([l*(l+1) for l in range(L+1)],float)
            mu=p@lam; sg=np.sqrt(max(p@lam**2-mu**2,0)); lo=int(np.argmin(abs(lam-(mu-sg)))); hi=int(np.argmin(abs(lam-(mu+sg))))
            sel.append((lo,hi)); add('mu',mu); add('sigma',sg); add('frac',((2*lo+1)+(2*hi+1 if hi!=lo else 0))/49)
            for (a,b) in SETS:
                msk=(deg==a)|(deg==b); v=auc(F[tr][:,msk],y[tr],F[te][:,msk],y[te]); add(f'set {a},{b}',v)
                if (a,b)==(lo,hi): add('selected',v)
            add('all49',auc(F[tr],y[tr],F[te],y[te])); add('ch59',auc(Fch[tr],y[tr],Fch[te],y[te]))
            c=CSP(n_components=8).fit(X[tr],y[tr]); add('csp',auc(c.transform(X[tr]),y[tr],c.transform(X[te]),y[te]))
            cv=Covariances('oas'); ts=TangentSpace(metric='riemann'); a1=ts.fit_transform(cv.transform(X[tr])); a2=ts.transform(cv.transform(X[te]))
            add('tangent',roc_auc_score(y[te],LogisticRegression(C=1.0,max_iter=1000).fit(a1,y[tr]).decision_function(a2)))
    r={kk:float(np.mean(v)) for kk,v in R.items()}; 
    from collections import Counter
    r['selected_degrees']=Counter(sel).most_common(3); r['geom']=(round(float(k),4),round(float(res),4),nfit); r['classes']=[str(c[0]) for c in nfo['classes'].ravel()]
    r['p_l']=[round(float(v),4) for v in np.array([V[:,deg==l].sum() for l in range(L+1)])/V.sum()]
    out[s]=r
    ranks=sorted([r[f'set {a},{b}'] for a,b in SETS],reverse=True)
    print(s,r['classes'],'geom',r['geom'],'p_l',r['p_l'],'\n   mu %.2f sigma %.2f selected %s frac %.3f | selected %.3f  rank %d/28  best set %.3f median set %.3f | set2,5 %.3f all49 %.3f ch59 %.3f csp %.3f tangent %.3f'%(
        r['mu'],r['sigma'],r['selected_degrees'],r['frac'],r['selected'],1+sum(v>r['selected'] for v in ranks),ranks[0],ranks[14],r['set 2,5'],r['all49'],r['ch59'],r['csp'],r['tangent']),flush=True)
json.dump(out,open('results_sphere.json','w'),indent=1,default=str)
H='abfg'; T=np.mean([out[s]['selected'] for s in H])
tab=[[out[s][f'set {a},{b}'] for a,b in SETS] for s in H]
null=np.array([np.mean(c) for c in itertools.product(*tab)])
print('\nPRIMARY: T = %.4f   null mean %.4f   p = %.4f   %s'%(T,null.mean(),(null>=T-1e-12).mean(),'SUPPORTED' if (null>=T-1e-12).mean()<0.05 else 'not supported'))
for kk in ['selected','set 2,5','all49','ch59','csp','tangent']: print('  human mean %-9s %.3f   artificial mean %.3f'%(kk,np.mean([out[s][kk] for s in H]),np.mean([out[s][kk] for s in 'cde'])))
