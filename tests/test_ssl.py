import numpy as np

from clinical_signal_copilot.ssl.contrastive import ContrastiveConfig, ContrastivePretrainer, nt_xent_loss
from clinical_signal_copilot.ssl.views import augment_pair
from clinical_signal_copilot.ssl.encoder import TemporalEncoder, featurize_windows


def test_augment_pair_shape(windows):
    X, _, _, _ = windows
    rng = np.random.default_rng(0)
    a, b = augment_pair(X[0], rng)
    assert a.shape == X[0].shape == b.shape


def test_featurize(windows):
    X, _, _, _ = windows
    F = featurize_windows(X[:5])
    assert F.shape[0] == 5
    assert F.ndim == 2
    assert np.isfinite(F).all()


def test_nt_xent_grads():
    rng = np.random.default_rng(0)
    z1 = rng.normal(size=(8, 4))
    z1 /= np.linalg.norm(z1, axis=1, keepdims=True)
    z2 = rng.normal(size=(8, 4))
    z2 /= np.linalg.norm(z2, axis=1, keepdims=True)
    loss, g1, g2 = nt_xent_loss(z1, z2, temperature=0.2)
    assert np.isfinite(loss)
    assert g1.shape == z1.shape


def test_contrastive_fit_transform(windows):
    X, _, _, _ = windows
    pre = ContrastivePretrainer(
        ContrastiveConfig(epochs=2, batch_size=16, max_windows=64, seed=1)
    )
    Z = pre.fit_transform(X)
    assert Z.shape[0] == len(X)
    assert Z.shape[1] == 32
    assert np.isfinite(Z).all()
    # embeddings roughly unit-norm
    norms = np.linalg.norm(Z, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_encoder_sgd_step():
    enc = TemporalEncoder(10, hidden=8, out_dim=4, seed=0)
    x = np.random.default_rng(0).normal(size=(5, 10))
    z, cache = enc.forward(x)
    enc.sgd_step(cache, np.ones_like(z) * 0.01, lr=1e-2)
    z2 = enc.encode(x)
    assert z2.shape == z.shape
