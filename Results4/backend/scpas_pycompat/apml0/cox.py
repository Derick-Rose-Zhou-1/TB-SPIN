"""Cox compatibility is intentionally not claimed in version 0.2.0.0."""

def fit(*args, **kwargs):
    raise NotImplementedError("COMPATIBILITY_PROFILE=BINOMIAL_ONLY; Cox is not validated")
