"""Exact SVD coefficients with a regularized first-order backward pass."""

import math

import torch
from torch.autograd.function import once_differentiable


class _CompressedSVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, rank, relative_gap):
        u, strengths, right = torch.linalg.svd(matrix, full_matrices = False)

        # Paired sign changes preserve the matrix. Use a reproducible right-side
        # convention; this does not remove rotations inside a repeated subspace.
        pivots = right.abs().argmax(dim = 1, keepdim = True)
        signs = torch.where(right.gather(1, pivots) < 0, -1, 1)
        u = u * signs.transpose(0, 1)
        right = right * signs

        ctx.rank = rank
        ctx.relative_gap = relative_gap
        ctx.save_for_backward(u, strengths, right)

        return strengths[:rank, None] * right[:rank]

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        u, strengths, right = ctx.saved_tensors
        rank = ctx.rank

        with torch.autocast(device_type = u.device.type, enabled = False):
            gradient = gradient.to(u.dtype)

            # C = U_r.T @ M. Only rotations involving a retained vector affect C;
            # discarded-to-discarded zero singular gaps never enter this backward.
            # Normalize strengths first: the scale cancels from these rotations,
            # avoiding overflow/underflow when forming squared singular gaps.
            normalized = strengths / strengths[0].clamp_min(torch.finfo(strengths.dtype).tiny)
            coordinates = normalized[:, None] * right
            gaps = normalized[:rank].square()[None] - normalized.square()[:, None]

            # An individual singular frame has no unique derivative at ties.
            # Bound near-tie rotations and choose zero rotation at an exact tie.
            # Away from this relative floor the ordinary SVD derivative is exact.
            inverse_gaps = gaps.sign() / gaps.abs().clamp_min(ctx.relative_gap)
            rotations = (gradient @ coordinates.transpose(0, 1)).transpose(0, 1) * inverse_gaps

            retained = u[:, :rank]
            matrix_gradient = retained @ gradient
            matrix_gradient = matrix_gradient + u @ (rotations @ coordinates[:rank])
            matrix_gradient = matrix_gradient + retained @ (rotations.transpose(0, 1) @ coordinates)

        return matrix_gradient, None, None


def compressed_svd(matrix, rank, relative_gap = 1e-6):
    """Return sign-canonicalized ``Sigma_r @ V_r.T`` for one valid matrix.

    The forward pass is the exact truncated SVD, with no input perturbation or
    reconstruction. The answer gradient reaches the matrix through both the
    retained values and their singular directions. Near repeated singular values,
    the direction gradient uses a bounded approximation because the exact
    derivative is not defined at ties. Higher-order derivatives are unsupported.

    Call this on each example's valid rows and columns before batch padding.
    Half-precision inputs are promoted to float32; float64 is preserved.
    """
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError('matrix must be a nonempty two-dimensional tensor')
    if not matrix.is_floating_point():
        raise TypeError('matrix must contain real floating-point values')
    if type(rank) is not int or rank <= 0:
        raise ValueError('rank must be a positive integer')
    if type(relative_gap) not in (int, float) or not math.isfinite(relative_gap) or relative_gap <= 0:
        raise ValueError('relative_gap must be a finite positive number')

    with torch.autocast(device_type = matrix.device.type, enabled = False):
        matrix = matrix if matrix.dtype in (torch.float32, torch.float64) else matrix.float()
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError('matrix must contain finite values')
        return _CompressedSVD.apply(matrix, min(rank, min(matrix.shape)), float(relative_gap))
