import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

from sklearn.mixture import GaussianMixture

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
          for n in n_components]
plt.figure()
plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');

values_AIC=[m.aic(X) for m in models];
values_BIC=[m.bic(X) for m in models];

min_AIC=values_AIC.index(min(values_AIC));
min_BIC=values_BIC.index(min(values_BIC));

gmm = GaussianMixture(n_components=min_AIC).fit(X) # does not work because min=3 != centers=4
labels = gmm.predict(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
