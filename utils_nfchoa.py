from scipy.special import hankel2, spherical_jn, spherical_yn
import numpy as np


def row_wise_norm2(array, p=2):
    r"""Compute the p-norm of every row of a 2D array.
    Parameters
    ----------
    array : numpy array
    p : float
        Order of the p-norm.
    """
    return np.sum(np.abs(array) ** p, axis=-1) ** (1. / p)


def spherical_hn2(n, z):
    r"""Spherical Hankel function of 2nd kind.
    Defined as https://dlmf.nist.gov/10.47.E6,
    .. math::
        \hankel{2}{n}{z} = \sqrt{\frac{\pi}{2z}}
        \Hankel{2}{n + \frac{1}{2}}{z},
    where :math:`\Hankel{2}{n}{\cdot}` is the Hankel function of the
    second kind and n-th order, and :math:`z` its complex argument.
    Parameters
    ----------
    n : array_like
        Order of the spherical Hankel function (n >= 0).
    z : array_like
        Argument of the spherical Hankel function.
    """
    return spherical_jn(n, z) - 1j * spherical_yn(n, z)


def max_order_circular_harmonics(N):
    r"""Maximum order of 2D/2.5D HOA.
    It returns the maximum order for which no spatial aliasing appears.
    It is given on page 132 of :cite:`Ahrens2012` as
    .. math::
        \mathtt{max\_order} =
            \begin{cases}
                N/2 - 1 & \text{even}\;N \\
                (N-1)/2 & \text{odd}\;N,
            \end{cases}
    which is equivalent to
    .. math::
        \mathtt{max\_order} = \big\lfloor \frac{N - 1}{2} \big\rfloor.
    Parameters
    ----------
    N : int
        Number of secondary sources.
    """
    return (N - 1) // 2


def nfchoa_25d(a_s, r_s, a_l, r_l, f, c=343):
    r"""Driving function of Near Field Compensated Ambisonics 2.5 D.
    Defined as with weighting as :cite:`Ahrens2012`.
    Parameters
    ----------
    a_s : float
          Angle of the primary source.
    r_s : float
          Distance to the origin of the primary source.
    a_l : array_like
          Angles of each of the secondary sources (loudspeakers).
    r_l : array_like
          Radius of secondary source (loudspeakers) circular arrangement.
    f : float
        (Time)-frequency.
    c : float
        Speed of light.
    """
    k = 2 * np.pi * f / c
    n_l = len(a_l)
    max_order = max_order_circular_harmonics(n_l)
    coefficients_indexes = range(-max_order, max_order + 1)
    coefficients = np.asarray([spherical_hn2(abs(m), k * r_s) / spherical_hn2(abs(m), k * r_l)
                               * np.e ** (1j * m * (a_l - a_s))
                               for m in coefficients_indexes])
    if r_s < r_l:
        limit = np.floor(k * r_s)
        weighting_vector = np.asarray([1 / 2 * (np.cos(m / limit * np.pi) + 1) if abs(m) <= limit else 0
                                       for m in coefficients_indexes])
        coefficients = (coefficients.T * weighting_vector).T
    return np.sum(coefficients, axis=0) / (2 * np.pi * r_l)


