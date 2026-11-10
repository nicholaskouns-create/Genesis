:- initialization(main).
w(S,C):-findall(1,(member(A,[-2,-1,0,1,2]),member(B,[-2,-1,0,1,2]),member(X,[-2,-1,0,1,2]),S=:=A+B+X),L),length(L,C).
main:-findall(D-Lm,(between(0,6,J),w(J,W1),J1 is J+1,w(J1,W2),D is (W1-W2)*(2*J+1),Lm is J*(J+1)),P),
 findall(D,member(D-_,P),Ds),sum_list(Ds,N),findall(X,(member(D-L,P),X is D*L),A),sum_list(A,S1),findall(X,(member(D-L,P),X is D*L*L),B),sum_list(B,S2),
 findall(D,(member(D-L,P),(N*L-S1)**2=:=N*S2-S1*S1),Kp),sum_list(Kp,Dim),Mu is S1 rdiv N,V is (N*S2-S1*S1) rdiv (N*N),
 format("prolog ~w N=~w mu=~w var=~w dim=~w~n",[Ds,N,Mu,V,Dim]),halt.
