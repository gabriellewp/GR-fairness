"""Reader-facing names; archived experiment identifiers remain unchanged."""


def method_label(method):
    if method == "GRID: ordinary continuation":
        return "GRID: standard fine-tuning"
    if method in ("GRID + IFairLRS RW (training)", "GRID + SPRINT (training)"):
        return method.replace("(training)", "(fairness-aware fine-tuning)")
    return method
