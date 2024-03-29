# Fast split selection by early features pruning
## Preface

Score = Score_L + Score_R

Score_L = SUM_L((f_L(x) - y) ** 2)

f_L(x) = Avg_L(x)


Assume `n` rows, `k` features.

Assume the following split check rounds
* `n / 100` rows, with `k` features
* n / 10,  k / 2,
* n / 4, k / 20,
* n, k / 50

The total running time would be:

(n / 100 * k + n / 10 * k / 2 + ...) << n * k

## Basic assumptions
Assume we sample n1 rows and find the optimal split of a feature.

The score of that feature is defined as

SUM_L((f_L(x) - y) ** 2) + 
SUM_R((f_R(x) - y) ** 2)


I should investigate deeply into this random variable - 
but roughly - while assuming the decision boundary is kept
the same when enlarging the sample - (i.e. L and R are constant sets).
Let n1_L the number of samples in the left sum.
Let n1_R the number of samples in the right sum.
We get that 

* SUM_L ~ Normal(n1_L * MU_L, n1_L * VARIANCE_L)
* SUM_R ~ Normal(n1_R * MU_R, n1_R * VARIANCE_R)

## Score analysis

So their sum, S = SUM_L + SUM_R, distributes

S ~ Normal(MU=n1_L * MU_L + n1_R * MU_R, 
           VARIANCE=n1_L * VARIANCE_L + n1_R * VARIANCE_R)


##  Pruning by random variables separation

### Approach 1: Naive separation
Assume we calculate the scores of two features: f1, f2

Calculate an estimates for 
VARIANCE_L1, VARIANCE_R1,  VARIANCE_L2, VARIANCE_R2
MU_L1, MU_R1, MU_L2, MU_R2

Then we calculate
S1 ~ Normal(MU1, VARIANCE1)
S2 ~ Normal(MU2, VARIANCE2)
respectively.

Therefore, if we can separate the random variables S1 and S2 -
we may prune the higher.
I.e., if we have

(MU1 - MU2) / sqrt(VARIANCE1 + VARIANCE2) >= 4 [or some other constant]

then we may prune the feature f1 because it is very unlikely to
produce the optimal split.
(And even it will - the difference is probably insignificant)


### Approach 2: Assumed variance, worst case latent expectation

Assume that we can estimate 

VARIANCE_L1, VARIANCE_R1,  VARIANCE_L2, VARIANCE_R2

exactly, and consider the worst case scenario for estimations of
 
MU_L1, MU_R1, MU_L2, MU_R2

#### Helper abstract problem
Let x1, x2, ... x_n be n samples of a normal variable 
with variance V.

What is the largest reasonable value for its expectancy ?
What is the smallest reasonable value for its expectancy ?

#### Helper problem solution
For each mu in the range [min(x_i), max(x_i)], calculate the likelihood
of the data, and calculate the reasonable range of likelihood of such data.

Log-likelihood of data:
CONST_A + CONST_B * SUM((mu - x_i) ** 2)
Reasonable range for the log likelihood:
    we can continue - but it get complicated and non-trivial to calculate




### Approach 3: Estimating the expectancy using t-distribution (the implemented approach) 
We have n1_L samples of Normal(MU_L, VARIANCE_L).
Using t-distribution, we can calculate easily what are
the reasonable values for MU_L.
Assume that it is between mu1_L and mu2_L with
confidence=95%.

By using equation 1 from 
https://accendoreliability.com/parameters-and-tolerance-estimates/

Denote the vector of samples as `x`

Denote the error `alpha = 1 - 95%`

`mu1_L = Avg(x) + t.ppf(alpha / 2, n1_L - 1) * Std(x) / Sqrt(n)`

`mu2_L = Avg(x) + t.ppf(1 - alpha / 2, n1_L - 1) * Std(x) / Sqrt(n)`


#### Minor improvement to approach 3
Actually, we don't have n1_L samples of Normal(MU_L, VARIANCE_L),
but n1_L samples of a random variable with expectancy MU_L
and variance VARIANCE_L.

One simple thing we can do in order to get samples of a normal variable,
is to partition the samples into sets of size ~= 30. The sum of 30 such
numbers indeed distributes normally. Then we have

n1_L / 30 samples of Normal(30 * MU_L, 30 * VARIANCE_L)

## Future directions
* Consider latent variance as well
* Do not consider only worst case evaluation of the expectancy
* Leverage the score form - i.e., leverage the fact that we have
f_L(x) = average_L(x), and then assume something on the distribution of y.
 
