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
