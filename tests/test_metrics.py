import numpy as np

from evaluate_cached_head import score


def test_score_returns_finite_challenge_metrics():
    labels = np.tile(np.array([[0, 1] * 20], dtype=np.float32), (8, 1))
    labels[1::2] = 1 - labels[1::2]
    probabilities = np.where(labels == 1, 0.9, 0.1)

    result = score(labels, probabilities)

    assert set(result) == {
        'mA', 'Label_F1', 'Inst_Acc', 'Inst_Prec', 'Inst_Rec', 'Inst_F1', 'Avg'
    }
    assert all(np.isfinite(value) for value in result.values())
    assert np.isclose(result['mA'], 1.0)
    assert np.isclose(result['Inst_F1'], 1.0)
    assert np.isclose(result['Avg'], 1.0)


def test_challenge_score_is_per_domain_harmonic_mean():
    from evaluate_cached_head import challenge_score
    labels = np.zeros((4, 40), dtype=np.float32)
    labels[:, 0] = [1, 0, 1, 0]
    labels[:, 1] = [0, 1, 0, 1]
    probs = labels.copy()
    probs[0, 0] = 0.0          # one miss in domain A only
    result = challenge_score(labels, probs, ['A', 'A', 'B', 'B'])
    a, b = result['domains']['A'], result['domains']['B']
    assert np.isclose(b['Challenge_Avg'], 1.0)
    # attributes 2..39 have no positives in A and are excluded: mA over attrs 0 and 1
    assert np.isclose(a["mA"], (0.5 + 1.0) / 2)
    expected = 2 * a['mA'] * a['Inst_F1'] / (a['mA'] + a['Inst_F1'])
    assert np.isclose(a['Challenge_Avg'], expected)
    assert np.isclose(result['Challenge_Avg'], (expected + 1.0) / 2)
