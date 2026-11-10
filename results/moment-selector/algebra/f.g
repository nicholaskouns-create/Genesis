w:=List([1..13],i->0);; for a in [-2..2] do for b in [-2..2] do for c in [-2..2] do w[a+b+c+7]:=w[a+b+c+7]+1; od; od; od;
d:=List([0..6],J->(w[J+7]-(function() if J+8<=13 then return w[J+8]; else return 0; fi; end)())*(2*J+1));; lam:=List([0..6],J->J*(J+1));;
N:=Sum(d);; mu:=Sum([1..7],i->d[i]*lam[i])/N;; v:=Sum([1..7],i->d[i]*lam[i]^2)/N-mu^2;;
Print("gap ",d," N=",N," mu=",mu," var=",v," dim=",Sum(Filtered([1..7],i->(lam[i]-mu)^2=v),i->d[i]),"\n"); QUIT;
