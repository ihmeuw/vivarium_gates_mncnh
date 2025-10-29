import numpy as np
import scipy.stats


def extract_upper_3(m):
    """Get the 3 elements in the upper triangle of
    a 3x3 matrix m.
    """
    m = np.asarray(m)
    return m[0, 1], m[0, 2], m[1, 2]


def symmetric_minor_determinant(p):
    return 1 - p**2


def asymmetric_minor_determinant(p12, p13, p23):
    return p12 * p23 - p13


def full_determinant(p1, p2, p3):
    return 1 + 2 * p1 * p2 * p3 - p1**2 - p2**2 - p3**2


def check_positive_semidefinite(p12, p13, p23, definite=False):
    """Check if three numbers can be the upper or lower
    triangle of a 3x3 correlation matrix, using Sylvester's
    criterion. The matrix is assumed to be symmetric with 1s
    on the diagonal.
    """
    leading_2x2_det = symmetric_minor_determinant(p12)
    if leading_2x2_det <= 0:
        if definite or leading_2x2_det < 0:
            return False
    leading_3x3_det = full_determinant(p12, p13, p23)
    if leading_3x3_det <= 0:
        if definite or leading_3x3_det < 0:
            return False
    if definite:
        # For definite matrices we only need to check
        # the 2 nontrivial leading principal minors
        return True

    # For semidefinite matrices, we need to check
    # the remaining three 2x2 principal minors
    for p in (p13, p23):
        if symmetric_minor_determinant(p) < 0:
            return False
    # # Oops, these minors are not _principal_
    # if asymmetric_minor_determinant(p12, p13, p23) < 0:
    #     return False

    return True


def generate_correlation_matrix(A):
    """Generates a correlation matrix (which is necessarily
    positive semidefinite) from an arbitrary matrix A
    by treating the matrix A @ A.T as a covariance matrix
    of some set of variables.
    """
    positive_semidef = A @ A.T
    std = np.diag(positive_semidef) ** (1 / 2)
    std_sq = std.reshape(1, -1) * std.reshape(-1, 1)
    corr = positive_semidef / std_sq
    return corr


def random_correlation_matrix(n_dim, dist=scipy.stats.norm(), seed=None):
    rng = np.random.default_rng(seed)
    random_matrix = dist.rvs(size=(n_dim, n_dim), random_state=rng)
    corr_matrix = generate_correlation_matrix(random_matrix)
    return corr_matrix
