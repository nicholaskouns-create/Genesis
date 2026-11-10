; the selected eigenvalues are EXACTLY 6 and 30: no Casimir value l=J(J+1), 0<=J<=6, other than those solves (l-18)^2=144,
; and n=3 is the only tensor order with mu-sigma = j(j+1), i.e. 3(n-1)=2n
(declare-const J Int)(declare-const n Int)
(push)(assert (and (>= J 0) (<= J 6) (= (* (- (* J (+ J 1)) 18) (- (* J (+ J 1)) 18)) 144) (not (= J 2)) (not (= J 5))))(check-sat)(pop)
(push)(assert (and (>= n 2) (= (* 3 (- n 1)) (* 2 n)) (not (= n 3))))(check-sat)(pop)
