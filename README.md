# The one-coordinate step in the Fourier entropy-influence conjecture

Yitzchak Shmalo

This repository contains a short note on the Fourier entropy-influence (FEI) conjecture of Friedgut and Kalai,
together with the scripts used for its numerical checks.

Paper: [main.pdf](main.pdf) (source [main.tex](main.tex)).

## Summary

For a Boolean function f, restricting one coordinate i lowers the Fourier entropy by
D_i = H(f) - (H(f+) + H(f-))/2. This step splits as D_i = E_i + J_i, where E_i is the conditional entropy
of the event "i is in U" given the rest of the spectral sample U, and J_i is a Jensen-Shannon divergence
(Gonzalez, MacManus and Pereyra, arXiv:2606.00246). The note proves:

* J_i <= (2/ln 2) Inf_i(f), with 2/ln 2 sharp;
* sum_i E_i(f) <= H(f);
* consequently FEI holds if and only if every nonconstant Boolean function has one coordinate with
  E_i <= C Inf_i, for some absolute C; the same holds with D_i in place of E_i. Writing C_n for the best FEI
  constant on n variables and R_n for the best such C, R_n <= C_n <= max_{m <= n} R_m + 2/ln 2;
* FEI is equivalent, up to an additive 2/ln 2 in the constant, to sum_i E_i <= C I(f), a pseudo-likelihood
  form;
* every coordinate of a function attaining C_n has influence at most 2^(3/ln 2 - C_(n-1));
* the condition cannot be required at every coordinate: replacing one point of a bent function on k
  variables by a fresh variable gives a coordinate with E_0/Inf_0 = k + log2 e + O(2^(-k/2)).

## Code

The scripts need Python 3 and NumPy. Each runs in a few seconds and writes nothing except to standard output;
the saved outputs are next to them.

* `code/check_erasure_indep.py` checks D_i = E_i + J_i on every Boolean function of four variables and on
  40,000 random functions each of five, six and seven variables at three densities. D_i is computed from the
  restricted truth tables, E_i and J_i from the Fourier coefficients. It also reports max J_i/Inf_i and
  max_f min_i E_i/Inf_i.
* `code/check_jsd_lemma.py` checks the inequality F(t) = 2 sin^2 t - phi(sin 2t) >= 0 behind Lemma 3 on a
  grid of two million points, the closed form of F', and the limit J/Inf -> 2/ln 2.
* `code/check_sec7.py` computes E_0/Inf_0 for the bent point-replacement functions of Proposition 7,
  k = 2, 4, ..., 10.

## License

The text is released under CC BY 4.0 and the code under the MIT license.
