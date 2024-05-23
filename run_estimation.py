import numpy as np
from sklearn.linear_model import LogisticRegression

from ConstrainedLogistic.model import ConstrainedLogisticRegression

np.random.seed(0)
n = 1000
x = np.random.normal(size=n*5).reshape((n, 5))
p = 1 / (1 + np.exp(-np.dot(x, np.array([1, 1, -1, -1, 0]))))
y = np.random.binomial(1, p, size=n)

# Fit the model with no constraints
clf = ConstrainedLogisticRegression(include_intercept=True)
clf.fit(x, y)
print(f'Results without constraints: {clf.coef}')


# Fit the model with non-negativity constraints
bounds = [(0, None) for _ in range(5)]
clf = ConstrainedLogisticRegression(include_intercept=True)
clf.fit(x, y, bounds=bounds)
print(f'Results with non-negativity constraints: {clf.coef}')

# compare results with SciKit-Learn
clf = LogisticRegression(fit_intercept=True, penalty=None)
clf.fit(x, y)
print(f'Results scikit-learn: {clf.coef_}')