"""Numerical check of Lemma 3 (J_i <= (2/ln 2) Inf_i).

Per spectral pair (a, b) = r(cos t, sin t), the Jensen-Shannon term is
r^2 (1 - h((1 + sin 2t)/2)) bits and the influence term is r^2 sin^2 t.
In nats the lemma is F(t) = 2 sin^2 t - phi(sin 2t) >= 0 with
phi(s) = ((1+s)ln(1+s) + (1-s)ln(1-s))/2.  We check F >= 0 on a fine grid of
[0, pi/2], the formula F'(t) = 2(sinh y - y)/cosh y with s = tanh y on
[0, pi/4), and that the ratio phi(sin 2t)/(ln 2 sin^2 t) tends to 2/ln 2.
"""
import numpy as np


def phi(s):
    # Near s = 0 use s*artanh(s) + log(1 - s^2)/2, which keeps full relative
    # precision; elsewhere the defining formula, which is accurate near s = 1.
    s = np.abs(np.asarray(s, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        small = s * np.arctanh(np.minimum(s, 0.5)) + 0.5 * np.log1p(-np.minimum(s, 0.5) ** 2)
        b = np.maximum(1 - s, 0.0)
        tb = np.where(b > 0, b * np.log(np.where(b > 0, b, 1.0)), 0.0)
        large = 0.5 * ((1 + s) * np.log1p(s) + tb)
    return np.where(s <= 0.5, small, large)


t = np.linspace(0.0, np.pi / 2, 2_000_001)
F = 2 * np.sin(t) ** 2 - phi(np.sin(2 * t))
print(f"min F on [0, pi/2] (2,000,001 points): {F.min():.3e}")

# derivative formula on [0, pi/4)
tt = np.linspace(1e-4, np.pi / 4 - 1e-4, 20_001)
s = np.sin(2 * tt)
y = np.arctanh(s)
closed = 2 * (np.sinh(y) - y) / np.cosh(y)
h = 1e-6
numeric = ((2 * np.sin(tt + h) ** 2 - phi(np.sin(2 * (tt + h))))
           - (2 * np.sin(tt - h) ** 2 - phi(np.sin(2 * (tt - h))))) / (2 * h)
print(f"max |F' - 2(sinh y - y)/cosh y| on [1e-4, pi/4 - 1e-4]: {np.abs(closed - numeric).max():.3e}")

# sharpness: J/Inf in bits -> 2/ln 2 as t -> 0
for tk in [1e-1, 1e-2, 1e-3]:
    jsd_bits = phi(np.sin(2 * tk)) / np.log(2)
    print(f"t={tk:g}: J/Inf = {jsd_bits / np.sin(tk) ** 2:.6f}   (2/ln 2 = {2 / np.log(2):.6f})")
