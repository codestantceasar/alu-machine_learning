\# Probability



\## Description



This project focuses on probability distributions using Python.



The main objective is to implement different probability distribution classes and understand how mathematical concepts such as expected value, probability mass functions, and probability density functions are applied in machine learning.



\## Files



\### poisson.py



Contains the `Poisson` class that represents a Poisson distribution.



The class can:



\- Estimate the expected number of occurrences (`lambtha`) from a dataset.

\- Use a provided lambda value when data is not available.

\- Calculate probabilities using Poisson distribution formulas.



\## Poisson Distribution



A Poisson distribution models the number of times an event occurs within a fixed interval of time or space.



The probability mass function is:



\\\[

P(X=k)=\\frac{\\lambda^k e^{-\\lambda}}{k!}

\\]



Where:



\- `λ` (lambtha) is the expected number of occurrences.

\- `k` is the number of occurrences.

\- `e` is Euler's number.



\## Requirements



\- Python 3

\- NumPy



\## Usage Example



```python

import numpy as np

from poisson import Poisson



np.random.seed(0)



data = np.random.poisson(5., 100).tolist()



p = Poisson(data)



print(p.lambtha)

