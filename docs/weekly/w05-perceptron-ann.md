# W05 Perceptron, Delta Rule & Neural Networks

**Question I am trying to answer:** Why can a linear model not learn XOR, and how does an MLP fix it?

**Initial hypothesis:** My answer to why XOR fails was "Cannot separate it using a straight line". For the perceptron update, I thought "Weights only change if errors". For the perceptron rule versus the delta rule, I only had "by threshold" and did not finish the idea. I thought non-linearity lets an MLP learn complex patterns, so it can learn XOR. For training, my answer was "Chain rule/backprop: gradient descent update from output to input".

<!-- Start with "Question I am trying to answer" and your 2-5 sentence initial hypothesis, written before you
open any AI tool (assignment Section 12.1, phase 0). W08 needs only a compact midterm-preparation entry.
Comments like this one are not counted as words. -->

## A. Concept capsule

The perceptron predicts `ŷ=sign(w·x+b)`. It updates only when `y(w·x+b)<=0`: `w←w+η·y·x`, `b←b+η·y`. The delta rule uses the output before the threshold, `o=w·x+b`, and updates on every sample with `η(t-o)x`. It is gradient descent on MSE. An MLP needs a nonlinear activation; without it all layers collapse into one linear map. Backprop only computes the gradient. SGD is a separate update step.

## B. One derivation or worked example

Take `w=(0,0)`, `b=0`, `η=1`, and one sample `x=(1,2)` with `y=+1`.

1. Score: `s=w·x+b=0`.
2. `y·s=0<=0`, so it is an error and we update.
3. `w=(0,0)+1·(+1)·(1,2)=(1,2)` and `b=0+1·(+1)=1`.
4. New score: `s'=1+4+1=6`.

The score rose by `η(‖x‖²+1)=6`, which matches `y·s'=y·s+η(‖x‖²+1)`.

## C. Code-to-theory trace

- `perceptron.py`, `perceptron_update`: the test `y*(w@x+b)<=0` is the error condition; the two assignments are the `w` and `b` updates.
- `mlp.py`, `backward`: `dZ=(P-Y)/n` is the gradient of softmax with mean cross-entropy; `A[l-1].T@dZ` gives `dW_l`; `(dZ@W_l.T)*(1-A[l-1]**2)` is the tanh derivative step.
- `mlp.py`, `forward`: the returned `cache` keeps the activations that `backward` needs.

## D. One controlled experiment

I ran my perceptron on AND and XOR with `η=1` and 20 epochs. The samples are visited in a fixed order, so there is no seed. I used a fixed loop calling `perceptron_update`. Mistakes per epoch:

- AND: 2, 3, 3, 2, 2, 3, 2, 1, then 0 from epoch 9 on.
- XOR: 4 in all 20 epochs; it never reaches 0.

## E. Failure / misconception

In the drill I wrote "Chain rule/backprop: gradient descent update from output to input". This mixes two steps. Backprop only gives `∇θL`; the update `θ←θ-η∇θL` is separate. In code, `loss` raised `UnboundLocalError` because it used `P` before calling `forward`.

## F. Written-exam capsule

**Question:** Prove that XOR cannot be separated by a line.

**Answer:** XOR needs `b<0` for (0,0), `w1+b>0` for (1,0), `w2+b>0` for (0,1), and `w1+w2+b<0` for (1,1). Adding the middle two gives `w1+w2+2b>0`, so `w1+w2+b>-b>0`. This contradicts the last condition. An MLP fixes this with a hidden layer and a nonlinear activation.

## G. Reflection

Before, I mixed up backprop and the update step; now I can explain both, and why XOR fails, with formulas. My weakest part was the perceptron rule versus the delta rule: I only had "by threshold". I am not sure I can write the backward pass closed-book. I will rewrite `backward` from a blank page before the next weekly entry.

## H. Inquiry trail

- Question I asked the AI: what `best_threshold`, `perceptron_update`, `forward`, `loss` and `backward` must compute, and why tests failed.
- Pre-AI attempt: drill `62ff803`. The code commits `e476b45`, `7e79e52`, `1c16920` came after debugging hints, so they are not pre-AI evidence.
- Hint summary: pointers to test errors (`~mask`, old `impurity` call, `loss` missing `forward`) and a review of my corrections file (wrong page numbers, leftover draft note). I wrote and fixed everything myself.
- Sources: `MLP.pdf`, `TrainingANN.pdf`, Mitchell (1997), Chapter 4.
- Without AI: update rule, worked example, XOR proof; not yet `backward`.

<!-- Mandatory whenever AI was used: the question you asked, the link to your pre-AI attempt commit, a summary of the
hint (not a pasted solution), the textbook/MOOC/repository source you used to verify it, and what you can now
reproduce without AI. For NPTEL videos use the 3-2-1 routine (Section 12.4) and link one exam-style question. -->
