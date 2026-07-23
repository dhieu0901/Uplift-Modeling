from pathlib import Path

from src.data.hillstrom import load_hillstrom, make_binary_hillstrom


def test_make_binary_hillstrom_keeps_treatment_and_control_only():
    path = Path("data/hillstrom_email.csv")
    if not path.exists():
        return

    raw = load_hillstrom(path)
    dataset = make_binary_hillstrom(raw, treatment_segment="Mens E-Mail", control_segment="No E-Mail")

    assert set(dataset.raw["segment"].unique()) == {"Mens E-Mail", "No E-Mail"}
    assert set(dataset.treatment.unique()) == {0, 1}
    assert len(dataset.X) == len(dataset.y) == len(dataset.treatment)
