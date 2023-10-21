from src.ops import *

class Group:
    def __init__(self, data, op, abel: bool = False, lie: bool = True) -> None:
        self.data = data
        self._op = op
        self.abel = abel
        self.lie = lie

    def __mul__(self, other):
        return ops_dict_[self._op](self, other)

    def commutator(self, other):
        return (self * other).__sub__(other * self)

    @staticmethod
    def character():
        pass