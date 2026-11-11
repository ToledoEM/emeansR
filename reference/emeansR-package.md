# emeansR: Mini-Batch k-Means Clustering with Exponential Moving Averages

Implements e-means (Bessa et al. (2026)
[doi:10.48550/arXiv.2611.09501](https://doi.org/10.48550/arXiv.2611.09501)
), a mini-batch replacement for Lloyd's k-means that updates centroids
with an exponential moving average and sets batch size and step size
from fitted scaling laws, following the reference 'faiss'
implementation. Reaches a lower loss than 25 Lloyd iterations in a few
epochs. The compute core is written in 'Rust'.

## See also

Useful links:

- <https://toledoem.github.io/emeansR/>

- <https://github.com/ToledoEM/emeansR>

- Report bugs at <https://github.com/ToledoEM/emeansR/issues>

## Author

**Maintainer**: Enrique Toledo <enriquetoledo@gmail.com>

Authors:

- Enrique Toledo <enriquetoledo@gmail.com>

Other contributors:

- Authors of the bundled Rust crates (see inst/AUTHORS) \[contributor,
  copyright holder\]
