Require Import ZArith List. Import ListNotations. Open Scope Z_scope.
Definition r := [-2;-1;0;1;2].
Definition w (s:Z) : Z := Z.of_nat (length (filter (fun t => Z.eqb t s) (flat_map (fun a => flat_map (fun b => map (fun c => a+b+c) r) r) r))).
Definition Js := [0;1;2;3;4;5;6].
Definition d (J:Z) := (w J - w (J+1)) * (2*J+1).
Definition lam (J:Z) := J*(J+1).
Definition sum l := fold_right Z.add 0 l.
Definition N := sum (map d Js).
Definition S1 := sum (map (fun J => d J * lam J) Js).
Definition S2 := sum (map (fun J => d J * lam J * lam J) Js).
Definition kept := filter (fun J => Z.eqb ((N*lam J - S1)*(N*lam J - S1)) (N*S2 - S1*S1)) Js.
Theorem census : map d Js = [1;9;25;28;27;22;13]. Proof. vm_compute. reflexivity. Qed.
Theorem carrier : N = 125. Proof. vm_compute. reflexivity. Qed.
Theorem mean : S1 = 18 * N. Proof. vm_compute. reflexivity. Qed.
Theorem variance : N*S2 - S1*S1 = 144 * N * N. Proof. vm_compute. reflexivity. Qed.
Theorem selected : kept = [2;5]. Proof. vm_compute. reflexivity. Qed.
Theorem dim47 : sum (map d kept) = 47. Proof. vm_compute. reflexivity. Qed.
Theorem faithful : forall J, In J Js -> ~ In J kept -> 4 * Z.abs ((lam J - 18)*(lam J - 18) - 144) >= 3 * 144.
Proof. intros J H; simpl in H; repeat (destruct H as [<-|H]; [vm_compute; intuition congruence|]); contradiction. Qed.
Print Assumptions dim47.
