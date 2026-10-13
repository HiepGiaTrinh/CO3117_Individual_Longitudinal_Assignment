# W06 — Bayesian Learning and Naive Bayes

BL = `BayesianLearning.pdf`, NB = `NaiveBayes.pdf`.

## Question I am trying to answer


**Question:** When data is small, how do the prior (MAP vs MLE) and the independence assumption (Naive Bayes) change a classifier's prediction?

**Hypothesis (from my first attempt, before AI):** I thought MAP differs from MLE mainly on small data because the prior weighs more (Q1c). In Q3c I guessed they agreed "because the prior is low". For the smoothed Naive Bayes result (Q4e) and for why it works with dependent features (Q5b) I wrote "Not sure".

## A. Concept capsule

**MLE vs MAP.** MLE maximises P(D|θ); MAP maximises P(D|θ)P(θ), so MLE is MAP with a uniform prior (BL p.12–14). For h=7, t=3, MLE = 0.70 but MAP with Beta(2,2) ≈ 0.67; with more data they converge (BL p.15, p.21–22).

**Naive Bayes.** It assumes the features are conditionally independent given the class, so P(C, X) = P(C) ∏ P(X_i | C) (NB p.3, p.5).

## B. Worked example by hand

Drill Q4, instance (Overcast, Strong).
- Unsmoothed: Yes = 5/8 × 1/5 × 1/5 = 0.025; No = 3/8 × 0/3 × 3/3 = 0, so Yes.
- Laplace: Yes = 5/8 × 2/8 × 2/7 ≈ 0.0446; No = 3/8 × 1/6 × 4/5 = 0.0500, so **No**.

## C. Code-to-theory trace

The code is in `src/from_scratch/naive_bayes.py` (commit `e42ae54`): a from-scratch `MultinomialNB` with Lidstone smoothing. It implements the decision rule `argmax P(C) ∏ P(x_i | C)` (NB p.11) in log space (log-prior plus `X @ feature_log_prob_.T`, normalised with log-sum-exp). Unlike the drill's categorical model, its denominator adds α × d (d = feature columns) instead of α × the number of values of each feature, so it does not reproduce 0.0446 and 0.0500.

## D. Controlled experiment: Laplace smoothing strength (not commited)

| α | Yes | No | Prediction |
|---|---|---|---|
| 0 | 0.0250 | 0.0000 | Yes |
| 1 | 0.0446 | 0.0500 | No |
| 5 | 0.0750 | 0.0641 | Yes |

The result is fragile in α, and 8 rows cannot say which α is better.

​```python
data=[("Sunny","Weak","Yes"),("Sunny","Strong","No"),("Rain","Weak","Yes"),("Rain","Strong","No"),
("Sunny","Weak","Yes"),("Rain","Weak","Yes"),("Overcast","Strong","Yes"),("Rain","Strong","No")]
V=[3,2]
def scores(train,x,a):
    out={}
    for c in ("Yes","No"):
        rows=[r for r in train if r[2]==c]; n=len(rows)
        s=n/len(train)
        for i,v in enumerate(x):
            cnt=sum(1 for r in rows if r[i]==v)
            s*=(cnt+a)/(n+a*V[i])
        out[c]=s
    return out
for a in (0,1,5):
    s=scores(data,("Overcast","Strong"),a)
    print(a,s,max(s,key=s.get))
​```

## E. Failure / misconception

**Q3c.** I said h_ML and h_MAP agree "because the prior is low". They agree because h3's likelihood advantage over h1 (15×) outweighs h1's prior advantage (7×).

**Q5a.** I wrote only "1 prior parameter" and forgot the P(X_i | C) tables. The count is 2n + 1 (NB p.6–7).

## F. Written-exam capsule

**Q (4 marks):** Why can Naive Bayes give a class score of exactly 0, and how does Laplace smoothing fix it?

**A:** A value never seen with a class gives P(x_i | C) = 0, so the whole product is 0 (NB p.11, p.33). Adding 1 to each count keeps every likelihood positive; in the drill No goes from 0 to 0.0500 and the prediction flips.

## G. Reflection


- **Can now explain:** why MLE is MAP with a uniform prior, and why one zero count kills a Naive Bayes score.
- **Not sure yet:** why Naive Bayes works with dependent features (Q5b); my answer is a heuristic I have not shown.
- **Will test next:** sweep α on the 14-row weather dataset (NB p.25) and duplicate a feature to break independence.

## H. Inquiry trail

- **Asked the AI:** draft the W06 drill (Q1–Q6); review `exercises/w06-corrections.md`; fix the grammar and wording of this file; AI compared my MultinomialNB with sklearn on the contract-test data before I committed it.
- Pre-AI attempt: drill `01f52a7` (handwritten, closed book, committed before any AI review); the sklearn comparison led to no code change.
- **AI suggested:** the structure of the drill questions; wording fixes only, with no change to my content.
- **Verified against:** BL p.8, p.12–15, p.20–22; NB p.3, p.5–7, p.11, p.33. Laplace smoothing is not in the slides; the rule is from the question.
- **Without AI:** Q1a–b, Q2, Q3a–b, Q4b, the unsmoothed scores in Q4c, Q6. Not without AI: Q4e, Q5b, Q5a.