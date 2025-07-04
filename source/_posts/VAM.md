# Approximate Distribution: ELBO

\( q_{\theta}(z) \approx p(z|x) \)

\( D(q_{\theta}(z) \| p(z|x)) = \mathbb{E}_{z \sim q} [\log \frac{q_{\theta}(z)}{p(z|x)}] = \mathbb{E}_{z \sim q} [\log q_{\theta}(z) - \log p(z|x)] \)

\( = \mathbb{E}_{z \sim q} [\log q_{\theta}(z) - \log \frac{p(z, x)}{p(x)}] \)

\( = \mathbb{E}_{z \sim q} [\log q_{\theta}(z) - \log p(z, x)] + \log p(x) \)

**evidence lower bound (elbo)**

\( \log p(x) = \mathbb{E}_{z \sim q} [\log p(z, x) - \log q_{\theta}(z)] + D(q_{\theta}(z) \| p(z|x)) \)

\( \log p(x) \geq \mathbb{E}_{z \sim q} [\log p(z, x) - \log q_{\theta}(z)] \equiv \mathcal{L}_q \)

**objective: minimizing kl divergence → maximizing elbo** (variational optimization: optimizing over functions)