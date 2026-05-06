"""
training.py: Training Network Decoder Matrices from Activation Data
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch import Tensor
import numba

if TYPE_CHECKING:
    pass

__author__ = "Chris Fritz"
__email__ = "fritz17236@hotmail.com"


from enum import Enum
import multiprocessing as mp
import os
import tqdm
import numpy as np
from numpy.typing import NDArray
from . import params


class RegressorType(Enum):
    HUBER = None


def _regress_neuron(
    acts: NDArray[np.floating], target_outputs: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Regress a given neuron's activations onto the provided target output.

    Args:
        acts (NDArray[np.floating]): The neuron's activations for the provied sample data, having shape [NUM_SAMPLES] x [DEPTH]
        target_outputs (NDArray[np.floating]): The target outputs the neuron should be mapped to having shape [NUM_SAMPLES] x [OUTPUT_DIM]

    Returns:
        NDArray[np.floating]: The regression coefficients that map the neuron's output to target output having shape [DEPTH] x [OUTPUT_DIM]
    """
    print(f"{acts.shape=}; {target_outputs.shape=}")
    return np.linalg.lstsq(acts, target_outputs, rcond=None)[0]


def compute_decoders(
    activations: NDArray[np.floating],
    target_output: NDArray[np.floating],
    config: params.RegressionParams,
) -> NDArray[np.floating]:
    """Compute the decoders that map activations associated with target input to the target output

    Args:
        activations (NDArray[np.floating]): Activations associated with target input having shape [DEPTH] x [WIDTH] x [NUM_SAMPLES]
        target_output (NDArray[np.floating]): The target output to regress against, having shape [NUM_SAMPLES] x [OUTPUT_DIMENSION]
        config (params.RegressionParams): The regression configuration specifying, width, and depth.

    Returns:
        NDArray[np.floating]: Decoder matrix for each neuron having shape [DEPTH] x [OUTPUT_DIMENSION] x [WIDTH]
    """
    # sanity check activations and target output match what is contained in config
    act_depth, act_width, act_num_samples = activations.shape
    num_samples, output_dim = target_output.shape

    if not act_depth == config.depth:
        raise ValueError(
            f"Mismatch between provided activations with depth dimension (0) = {act_depth} and provided configuration depth = {config.depth}"
        )

    if not act_width == config.width:
        raise ValueError(
            f"Mismatch between provided activations with width dimension (1) = {act_width} and provided configuration width = {config.width}"
        )

    if not output_dim == config.output_dimension:
        raise ValueError(
            f"Mismatch between provided target output dimension with dimension (0) = {output_dim} and provided configuration's output dimension = {config.output_dimension}"
        )

    if not act_num_samples == num_samples:
        raise ValueError(
            f"Mismatch between provided activations with sample count dimension (2) = {act_depth} and target outputs with sample count {num_samples}"
        )

    # multiprocessing to perform regression on per-neuron basis
    cpu_count = mp.cpu_count()
    if len(os.sched_getaffinity(0)) < cpu_count:
        try:
            os.sched_setaffinity(0, range(cpu_count))
        except OSError:
            print("Could not set affinity")
    num_worker_procs = len(os.sched_getaffinity(0))

    with mp.Pool(num_worker_procs) as p:
        decoders = np.stack(
            *[
                p.starmap(
                    func=_regress_neuron,
                    iterable=tqdm.tqdm(
                        [
                            (
                                activations[:, idx_neuron, :].T,
                                target_output,
                            )
                            for idx_neuron in range(config.width)
                        ],
                        total=config.width,
                    ),
                )
            ],
            axis=-1,
        )

    # confirm correct shape output
    dim0, dim1, dim2 = decoders.shape
    if not dim0 == act_depth:
        raise RuntimeError(
            f"Computed decoder matrix has invalid shape: Target depth dimension = {act_depth}; dim0 of decoder matrix = {dim0}"
        )

    if not dim1 == output_dim:
        raise RuntimeError(
            f"Computed decoder matrix has invalid shape: Target output dimension = {output_dim}; dim1 of decoder matrix = {dim1}"
        )

    if not dim2 == act_width:
        raise RuntimeError(
            f"Computed decoder matrix has invalid shape: Target width dimension = {act_width}; dim2 of decoder matrix = {dim2}"
        )

    return decoders

# =========================
# JIT CORE (no Python here)
# =========================

@numba.njit(fastmath=True)
def _update(W, P, a, b, lam):
    d = W.shape[0]
    m = W.shape[1]

    # Pa = P @ a
    Pa = P @ a

    # denom = lam + a^T P a
    denom = lam
    for i in range(d):
        denom += a[i] * Pa[i]

    # K = Pa / denom
    K = Pa / denom

    # pred = a^T W
    pred = np.zeros(m)
    for j in range(m):
        s = 0.0
        for i in range(d):
            s += a[i] * W[i, j]
        pred[j] = s

    # err = b - pred
    err = np.empty(m)
    for j in range(m):
        err[j] = b[j] - pred[j]

    # W += K outer err
    for i in range(d):
        ki = K[i]
        for j in range(m):
            W[i, j] += ki * err[j]

    # compute a^T P
    aTP = np.zeros(d)
    for j in range(d):
        s = 0.0
        for i in range(d):
            s += a[i] * P[i, j]
        aTP[j] = s

    # P update
    for i in range(d):
        for j in range(d):
            P[i, j] = (P[i, j] - K[i] * aTP[j]) / lam


@numba.njit(fastmath=True)
def _fit(W, P, A, B, lam):
    n = A.shape[0]
    for i in range(n):
        _update(W, P, A[i], B[i], lam)



class RLS:
    def __init__(self, dim_in: int, dim_out: int, lam: float = 1.0, delta: float = 1e5):
        """Initialize a RLS Solver"""
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.lam = lam
        self.delta = delta

        # ⚡ float32 = big speed win
        self.W = np.zeros((dim_in, dim_out), dtype=np.float32)
        self.P = (delta * np.eye(dim_in)).astype(np.float32)

    def update(self, a: np.ndarray, b: np.ndarray):
        """Single update (still fast via JIT core)"""
        a = np.asarray(a, dtype=self.W.dtype)
        b = np.asarray(b, dtype=self.W.dtype)

        if a.shape != (self.dim_in,):
            raise ValueError(f"{a.shape=} expected {(self.dim_in,)}")

        if b.shape != (self.dim_out,):
            raise ValueError(f"{b.shape=} expected {(self.dim_out,)}")

        _update(self.W, self.P, a, b, self.lam)

    def fit(self, A: np.ndarray, B: np.ndarray):
        """Fit with tqdm progress bar (Python loop, JIT inner)"""

        if A.ndim != 2:
            raise ValueError(f"{A.shape=} must be 2D")

        if B.ndim != 2:
            raise ValueError(f"{B.shape=} must be 2D")

        if A.shape[0] != B.shape[0]:
            raise ValueError("A and B must have same number of rows")

        A = np.asarray(A, dtype=self.W.dtype)
        B = np.asarray(B, dtype=self.W.dtype)

        num_obs = A.shape[0]
        print(f"Training on num_obs={num_obs}")

        # 🚀 keep tqdm, still fast because update is JIT
        for i in tqdm.tqdm(range(num_obs), desc="Fitting Estimator"):
            _update(self.W, self.P, A[i], B[i], self.lam)

        return self.W

    def fit_fast(self, A: np.ndarray, B: np.ndarray):
        """Maximum speed (no tqdm, fully JIT loop)"""
        A = np.asarray(A, dtype=self.W.dtype)
        B = np.asarray(B, dtype=self.W.dtype)

        _fit(self.W, self.P, A, B, self.lam)
        return self.W

    def predict(self, A):
        A = np.asarray(A, dtype=self.W.dtype)
        return A @ self.W
    
class BlockLowRankRLS:
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        rank: int = 64,
        lam: float = 1.0,
        delta: float = 1e2,
        device: str = "cuda",
        dtype=torch.float32,
        max_rank: int = 256,
        adapt_rank: bool = True,
    ):
        self.d = dim_in
        self.m = dim_out
        self.rank = rank
        self.max_rank = max_rank
        self.lam = lam
        self.device = device
        self.adapt_rank = adapt_rank

        # weights
        self.W = torch.zeros(dim_in, dim_out, device=device, dtype=dtype)

        # diagonal covariance
        self.D = torch.full((dim_in,), delta, device=device, dtype=dtype)

        # low-rank factor
        self.U = torch.zeros(dim_in, rank, device=device, dtype=dtype)

    def _apply_P(self, A: Tensor):
        """
        A: (batch, d)
        returns: (batch, d)
        """
        DA = A * self.D            # (b, d)
        UA = A @ self.U            # (b, k)
        return DA + UA @ self.U.T  # (b, d)

    def update_block(self, A: Tensor, B: Tensor):
        """
        A: (batch, d)
        B: (batch, m)
        """
        # ---- Apply covariance ----
        PA = self._apply_P(A)              # (b, d)

        # ---- Compute gain ----
        b = A.shape[0]

        S = self.lam * torch.eye(b, device=self.device) + A @ PA.T

        # jitter (critical)


        try:
            # preferred path
            X = torch.linalg.solve(S, PA)   # (b, d)
        except RuntimeError:
            # fallback if still singular
            X = torch.linalg.lstsq(S, PA).solution

        K = X.T  # (d, b)

        # ---- Prediction ----
        pred = A @ self.W                  # (b, m)
        err = B - pred                     # (b, m)

        # ---- Update weights ----
        self.W = self.W + K @ err          # (d, m)

        # ---- Update covariance (low-rank) ----
        # approximate: P <- (P - K A P) / lam

        # update diagonal
        APA = torch.sum(A * PA, dim=1)     # (b,)
        diag_update = (K * APA.unsqueeze(0)).sum(dim=1)
        self.D = (self.D - diag_update) / self.lam

        # update low-rank factors
        if self.rank > 0:
            self.U = torch.cat([self.U, K], dim=1)

            # ---- rank adaptation via energy ----
            if self.adapt_rank:
                self._truncate_rank()

    def _truncate_rank(self):
        """
        Keep top directions based on energy (QR-based)
        """
        # QR is much cheaper than full SVD
        Q, R = torch.linalg.qr(self.U)

        # energy per component
        energy = torch.sum(R**2, dim=1)

        # sort by energy
        idx = torch.argsort(energy, descending=True)

        k = min(self.max_rank, self.U.shape[1])
        self.U = Q[:, idx[:k]]

    def fit(self, A: Tensor, B: Tensor, batch_size: int = 128):
        n = A.shape[0]

        for i in range(0, n, batch_size):
            Ab = A[i:i+batch_size].to(self.device)
            Bb = B[i:i+batch_size].to(self.device)

            self.update_block(Ab, Bb)

        return self.W

    def predict(self, A: Tensor):
        return A.to(self.device) @ self.W