\# Bayesian Probability



\## Description



This project implements Bayesian probability concepts using Python and NumPy.



The project applies Bayes' theorem to calculate probabilities related to a medical study where patients take a drug and we estimate the probability of severe side effects.



The main concepts covered are:



\- Likelihood

\- Intersection probability

\- Marginal probability

\- Posterior probability



\## Files



\### 0-likelihood.py



Contains the `likelihood()` function.



Calculates the likelihood of observing `x` patients with severe side effects out of `n` patients for different hypothetical probabilities.



Formula:



\\\[

P(X=x|p)=\\binom{n}{x}p^x(1-p)^{n-x}

\\]





\### 1-intersection.py



Contains the `intersection()` function.



Calculates the intersection between the likelihood and prior probability.



Formula:



\\\[

P(X=x \\cap P)=P(X=x|P)P(P)

\\]





\### 2-marginal.py



Contains the `marginal()` function.



Calculates the total probability of obtaining the observed data.



Formula:



\\\[

P(X=x)=\\sum P(X=x \\cap P)

\\]





\### 3-posterior.py



Contains the `posterior()` function.



Uses Bayes' theorem to calculate updated probabilities after observing data.



Formula:



\\\[

P(P|X=x)=\\frac{P(X=x|P)P(P)}{P(X=x)}

\\]





\## Requirements



\- Python 3

\- NumPy



\## Usage



Example:



```python

import numpy as np



P = np.linspace(0, 1, 11)



result = likelihood(26, 130, P)



print(result)

