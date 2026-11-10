w=vector(13,i,0);for(a=-2,2,for(b=-2,2,for(c=-2,2,w[a+b+c+7]++)));
d=vector(7,J,(w[J+6]-if(J+7<=13,w[J+7],0))*(2*J-1));lam=vector(7,J,(J-1)*J);
N=vecsum(d);mu=sum(i=1,7,d[i]*lam[i])/N;v=sum(i=1,7,d[i]*lam[i]^2)/N-mu^2;
D=sum(i=1,7,if((lam[i]-mu)^2==v,d[i],0));print("pari-gp ",d," N=",N," mu=",mu," var=",v," roots=",mu-sqrtint(v),",",mu+sqrtint(v)," dim=",D);quit
