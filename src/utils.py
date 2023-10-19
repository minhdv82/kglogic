
def clip(value, lo=0, hi=1) -> float:
    return lo if value < lo else hi if value > hi else value