# W06 drill corrections

Page numbers are for the current version of `BayesianLearning.pdf` and `NaiveBayes.pdf`.

## Q1c. Why h_MAP can differ from h_ML on a small training set
- What I wrote: With a small training set, h_MAP depends more on the prior, so the posterior may differ from h_ML.
- What was wrong or missing: The idea is right, but it stops halfway. I did not say why the prior matters (the likelihood is weak with little data), and I left out the other direction: with a lot of data the two converge.
- Correct reasoning: With few samples, P(D|h) barely separates the hypotheses, so P(h) can change the argmax of P(D|h)P(h). As the sample grows, the likelihood dominates and h_MAP approaches h_ML.
- Source (slide, page): `BayesianLearning.pdf` p.15 (MLE vs MAP: scarce data, prior has stronger influence; abundant data, results are similar)

## Q3c. Why h_ML and h_MAP differ (or not) in the 3-hypothesis table
- What I wrote: They are the same in this case (h3), maybe because the prior is low.
- What was wrong or missing: The conclusion (same answer, h3) is right, but the reason is wrong and hedged. The prior is not "low": h1 has the largest prior (0.70). I also did not compare the prior ratio with the likelihood ratio.
- Correct reasoning: h_MAP maximizes P(D|h)P(h). h1 beats h3 on prior by 0.70/0.10 = 7x, but h3 beats h1 on likelihood by 0.30/0.02 = 15x. The likelihood gap is bigger, so the posterior still favors h3 (0.469 vs 0.219). The two estimators would only disagree if the prior ratio exceeded the likelihood ratio, e.g. P(h1) large enough that 0.02 * P(h1) > 0.30 * P(h3).
- Source (slide, page): `BayesianLearning.pdf` p.14 (MAP maximizes P(D|θ)P(θ); MLE ignores the prior) and p.15 (when the prior matters)

## Q4a. Naive Bayes decision rule and independence assumption
- What I wrote: ŷ = argmax_y P(y) ∏ P(x_i | y).
- What was wrong or missing: The decision rule is correct, but the question also asked me to state the independence assumption, and I did not.
- Correct reasoning: Naive Bayes assumes the features are conditionally independent given the class: X_i ⊥ X_-i | C. This is what lets the joint factor as P(C, X1..XD) = P(C) ∏ P(X_i | C), and the posterior become proportional to P(C) ∏ P(x_i | C).
- Source (slide, page): `NaiveBayes.pdf` p.3 (the assumption), p.5 (factored joint), p.11 (argmax rule)

## Q4c. Unnormalised scores for (Overcast, Strong)
- What I wrote: Yes: 5/8 · 1/5 · 1/5 = 0.025; No: 3/8 · 0 · 1 = 0 (the last factor looked like "0.1" in my handwriting); prediction is Yes.
- What was wrong or missing: The numbers are correct, but the No line is ambiguous. "0.1" reads as a probability of 0.1, whereas the factor I meant is P(Strong | No) = 3/3 = 1. I should show every factor in the same fraction form.
- Correct reasoning: Yes = 5/8 × 1/5 × 1/5 = 0.025. No = 3/8 × P(Overcast|No) × P(Strong|No) = 3/8 × 0/3 × 3/3 = 0. The unsmoothed prediction is Yes, but see Q4d for why this result cannot be trusted.
- Source (slide, page): `NaiveBayes.pdf` p.11 (product of prior and per-feature likelihoods), p.33 (the same kind of 0/5 = 0.0 entry for Overcast given No in the slide's dataset)

## Q4d. What goes wrong in (c) and why
- What I wrote: Because Overcast never appears with No in the data.
- What was wrong or missing: I only stated the cause. I did not explain the effect, namely that the zero kills the whole product and ignores the other evidence.
- Correct reasoning: P(Overcast | No) = 0/3 makes the No score exactly 0 no matter what the other factors are. Here Strong wind is 3/3 for No, which is strong evidence for No, yet it is ignored. A value that is merely unseen in a small sample is being treated as impossible, which is overfitting to the counts. Smoothing is needed.
- Source (slide, page): `NaiveBayes.pdf` p.32 (Overcast count is 0 when Play = No), p.33 (P(O | No) = 0/5 = 0.0), p.34 (this zero carried into the summary table)

## Q4e. Laplace smoothing and final class
- What I wrote: Not sure.
- What was wrong or missing: Nothing attempted. The slides do not cover Laplace smoothing itself; the rule is given in the question statement (add 1 to each count, add the number of attribute values to the denominator).
- Correct reasoning: Outlook has 3 values, Wind has 2.
  - Yes: P(Overcast|Yes) = (1+1)/(5+3) = 2/8; P(Strong|Yes) = (1+1)/(5+2) = 2/7. Score = 5/8 × 2/8 × 2/7 ≈ 0.0446.
  - No: P(Overcast|No) = (0+1)/(3+3) = 1/6; P(Strong|No) = (3+1)/(3+2) = 4/5. Score = 3/8 × 1/6 × 4/5 = 0.0500.
  - 0.0500 > 0.0446, so the final class is No. The zero no longer wipes out the Strong-wind evidence.
  - Link to the slides: for a binary variable, a Beta(2,2) prior gives the MAP estimate (h+1)/(N+2), which is Laplace smoothing; it is the same idea of adding pseudo-counts as a prior.
- Source (slide, page): `NaiveBayes.pdf` p.33 (the zero-count problem being fixed); `BayesianLearning.pdf` p.20 (MAP formula with prior counts) and p.21 (Beta(2,2) worked example)

## Q5a. Number of parameters
- What I wrote: Naive Bayes: 1 prior parameter. Full joint: 2^(n+1) − 1.
- What was wrong or missing: The full-joint count is correct. The Naive Bayes count is incomplete: I only counted the prior and left out the conditional probabilities P(x_i | y).
- Correct reasoning: For a binary class there is 1 free parameter for P(C). For each of the n binary attributes there is 1 free parameter per class value, P(x_i = 1 | C = c) for c in {0, 1}, which is 2n. Total = 2n + 1. This is linear in n, versus exponential (2^(n+1) − 1) for the full joint.
- Source (slide, page): `NaiveBayes.pdf` p.6 (model parameters are P(C) and P(X_i | C)), p.7 (tables of P(C) and P(X_i | C) for discrete variables)

## Q5b. A situation where independence is false, and why Naive Bayes still works
- What I wrote: Not sure.
- What was wrong or missing: Nothing attempted.
- Correct reasoning: Example: in spam filtering, the words "free" and "win" often occur together, so they are not independent given the class. Naive Bayes still classifies reasonably because it only needs the correct class to have the highest score, not exact probabilities. Dependencies tend to distort the scores of the competing classes in a similar way, so the argmax often stays the same. It also needs very few parameters, which is helpful with high-dimensional data like text.
- Source (slide, page): `NaiveBayes.pdf` p.4 (assumption called "naive" because it is too strong); `BayesianLearning.pdf` p.27 (independence rarely holds, yet Naive Bayes is simple and effective for high-dimensional data such as text)

## Q6. Why use a sum of logs instead of a product
- What I wrote: The product of many small probabilities becomes very small, so to avoid the underflow issue we use a sum of logs; the argmax is unchanged.
- What was wrong or missing: The answer is essentially correct and names underflow. Missing: why the argmax is unchanged (log is monotonic), and that a zero probability becomes log 0 = −∞ unless smoothed.
- Correct reasoning: Multiplying D small probabilities can fall below floating-point range and round to 0, making the classes impossible to compare. Since log is strictly increasing, argmax P(C) ∏ P(x_i|C) equals argmax [log P(C) + Σ log P(x_i|C)], and the sum of moderate negative numbers does not underflow.
- Source (slide, page): `NaiveBayes.pdf` p.11 (log form of the decision rule)
