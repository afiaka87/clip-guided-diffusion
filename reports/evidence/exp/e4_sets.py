# Confirmation on seeds the 128 comparison never saw, so its best variants cannot win on selection alone.
seeds = (2, 3)
SETS["confirm"] = {
    "base": Config(),
    "pure x0": Config(pure_x0=True),
    "disco schedule": Config(layout="disco"),
    "n16": Config(n=16),
    "pure x0 disco": Config(pure_x0=True, layout="disco"),
}
