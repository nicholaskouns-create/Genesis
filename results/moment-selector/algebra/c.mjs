const w=new Map();for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let c=-2;c<=2;c++){const s=a+b+c;w.set(s,(w.get(s)||0n)+1n)}
let d=[],lam=[];for(let J=0;J<=6;J++){d.push(((w.get(J)||0n)-(w.get(J+1)||0n))*BigInt(2*J+1));lam.push(BigInt(J*(J+1)))}
const N=d.reduce((a,b)=>a+b),S1=d.reduce((a,x,i)=>a+x*lam[i],0n),S2=d.reduce((a,x,i)=>a+x*lam[i]*lam[i],0n)
// (N*l - S1)^2 == N*S2 - S1^2  <=> (l-mu)^2 == var, in integers
const keep=lam.filter(l=>(N*l-S1)**2n===N*S2-S1*S1);const D=d.filter((x,i)=>keep.includes(lam[i])).reduce((a,b)=>a+b)
console.log("node-bigint",d.join(","),"N="+N,"mu="+S1/N,"var="+(N*S2-S1*S1)/(N*N),"keep="+keep,"dim="+D)
