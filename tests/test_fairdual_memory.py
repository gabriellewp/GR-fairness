from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fairdual_exposure_ranking_does_not_build_autograd_graphs():
    source = (ROOT / "fairdiverse/recommendation/trainer.py").read_text()
    fairdual_scoring = source.index(
        "_, indices = self.Model.full_ranking("
    )
    no_grad = source.rfind("with torch.no_grad():", 0, fairdual_scoring)

    assert no_grad != -1
    assert fairdual_scoring - no_grad < 500
