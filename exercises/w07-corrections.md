# W07 Corrections

Trinh Gia Hiep - 2352343

## Q1: Factorization
- What I wrote: P(S, P, R, W, L) = P(S) · P(P|S) · P(W|P,R) · P(L|W)
- What was wrong: I left out the factor for R. The product had only 4 factors, but the DAG has 5 nodes.
- Why: In a Bayesian network, every node contributes exactly one factor P(node | parents of node). R has S as its only parent (S → R), so it needs P(R|S). I wrote the factors for S, P, W, L and forgot R.
- Corrected: P(S, P, R, W, L) = P(S) · P(P|S) · P(R|S) · P(W|P,R) · P(L|W)
- Source: BayesNetIntroduction.pdf, slide 14.

## Q2: Is P ⟂ R | S? (d-separation)
- What I wrote: "Is P⊥R: Yes", and then I analyzed only the path P ← S → R.
- What was wrong: (a) I did not write the conditioning set. The statement P ⟂ R and the statement P ⟂ R | S are different claims, and the question asks about P ⟂ R | S. (b) There are 2 paths between P and R, but I analyzed only 1.
- Why: d-separation requires every path to be blocked. Path 1, P ← S → R, is a fork at S, and S is observed, so it is blocked. Path 2, P → W ← R, has a collider at W. A collider blocks the path unless W or one of its descendants is observed. Here W and its descendant L are both unobserved, so path 2 is blocked. All paths are blocked, so P ⟂ R | S holds.
- Corrected: P ⟂ R | S is true. Path 1 is blocked by the observed fork S. Path 2 is blocked by the unobserved collider W (and its descendant L is also unobserved).
- Source: Murphy (2022), Chapter 3

## Q4: Note (no correction needed)
- Q4 was correct. Two small points: in P(X2|C,X1), my X1 looked like X4 because of my handwriting, so I should write it more clearly. Also, a more precise statement is that TAN lets each feature have at most one extra feature parent besides the class, and these feature-to-feature edges form a tree. Naive Bayes is the special case with no such edges.