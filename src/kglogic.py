from typing import List, Union, Set

from .graph import *
from .ops import *
from .sat import Sat
from .utils import clip

ValueType = Union[float, Node]

# list of operators exposed to user
OPS_LIST = ('NOT', 'AND', 'OR', 'IMPL', 'EQUI', 'ONE')

class Symbol:
    def __init__(self, name: str, op: str, children: Set, value: ValueType = 1.) -> None:
        value = value if isinstance(value, Node) else Node(value)
        self._name = name
        self._op = op
        self._children = children
        self.truth = value

    def __repr__(self):
        tname = 'Atom' if self.is_atomic else 'Symbol'
        return f"<type: {tname}; name: {self._name}; truth: {self.truth.value}>"

    @property
    def is_atomic(self) -> bool:
        return not self._children

    def __not__(self):
        return Symbol(name='Not(' + self._name + ')', op='NOT',
                      children=(self,))

    def __or__(self, other):
        return Symbol(name='Or(' + self._name + ', ' + other._name + ')', op='OR',
                      children=(self, other))
        
    def __and__(self, other):
        return Symbol(name='And(' + self._name + ', ' + other._name + ')', op='AND',
                      children=(self, other))

    def __impl__(self, other):
        return Symbol(name='IMPL(' + self._name + ', ' + other._name + ')', op='IMPL',
                      children=(self, other))
    
    def __equi__(self, other):
        return Symbol(name='EQUI(' + self._name + ', ' + other._name + ')', op='EQUI',
                      children=(self, other))

    def eval(self):
        if self.is_atomic:
            return
        child_vals = []
        for child in self._children:
            child_vals.append(child.truth)
        self.truth = ops_dict_[self._op](*child_vals)

    # this runs in non-fuzzy mode
    def sat(self) -> bool:
        if self.is_atomic:
            return self.truth.value > .9
        child_vals = []
        for child in self._children:
            child_vals.append(child.truth.value > .9)
        self.truth.value = opsd_dict_[self._op](*child_vals)
        return self.truth.value > .9

    @property
    def requires_grad(self):
        return self.truth.requires_grad

    def none_val(self):
        self.truth.value = None

    def zero_grad(self):
        self.truth.grad = 0

    # get set of atoms that this symbol depends on
    def get_required_atoms(self):
        if self.is_atomic:
            return set()
        res = set()
        visited = set()
        def _recursive(symbol):
            if not symbol in visited:
                visited.add(symbol)
                if symbol.is_atomic:
                    res.add(symbol)
                    return
                for child in symbol._children:
                    _recursive(child)
        _recursive(self)

        return res

    def backward(self):
        self.truth.backward()

    def __hash__(self):
        return hash(self._name)
    
class KnowledgeBase:
    def __init__(self) -> None:
        self.symbols: List[Symbol] = []
        self.atoms: List[Symbol] = []
        self.topo: List[Symbol] = []

    def add(self, *args):
        for symbol in args:
            if symbol.is_atomic:
                self.atoms.append(symbol)
            else:
                self.symbols.append(symbol)
    
    def prepare_model(self, fuzzy: bool):
        import random
        for atom in self.atoms:
            atom.truth.value = random.uniform(0., 1.) if fuzzy else random.choice([0, 1])

    def eval(self):
        topo = self._build_topo()
        for x in topo:
            x.eval()

    def _print_atoms(self):
        for atom in self.atoms:
            print(atom)

    # find any model that verifies all the symbols
    def solve(self, fuzzy=False):
        is_solution = [False]
        def _sample(n_atoms, res, r):
            if n_atoms == 0:
                res.append(r)
                return
            for i in [0, 1]:
                _sample(n_atoms - 1, res, r + [i])

        def _recursive(unvisited_symbols, visited_atoms):
            if not unvisited_symbols:
                is_solution[0] = True
                print("solution")
                self._print_atoms()
                return
            symbol = unvisited_symbols[0]
            dep_atoms = symbol.get_atoms()
            atoms = dep_atoms - visited_atoms
            cnt = len(atoms)
            if cnt > 0:
                samples = []
                _sample(cnt, samples, [])
                for sample in samples:
                    for val, atom in zip(sample, atoms):
                        atom.truth.value = val
                    if symbol.sat() or symbol not in self.symbols:
                        _recursive(unvisited_symbols[1:], visited_atoms | atoms)
            else:
                if symbol.sat() or symbol not in self.symbols:
                    _recursive(unvisited_symbols[1:], visited_atoms)
        _recursive(self._build_topo(), set())
        if not is_solution[0]:
            print('No solution found!')

    def _build_topo(self):
        if self.topo:
            return self.topo
        topo = []
        visited = set()
        def _recursive(symbol: Symbol):
            if symbol.is_atomic or symbol in visited:
                return
            visited.add(symbol)
            for child in symbol._children:
                _recursive(child)
            topo.append(symbol)
        for symbol in self.symbols:
            _recursive(symbol)
        self.topo = topo
        return topo

    def entails(self, symbol: Symbol):
        atoms = symbol.get_required_atoms()

        pass

    def parameters(self):
        return [atom.truth for atom in self.atoms]

    def eval_mode(self):
        for param in self.parameters():
            param.requires_grad = False

    def train_mode(self):
        for param in self.parameters():
            param.requires_grad = True
            param.grad = 0

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def _round(self):
        for param in self.parameters():
            param.value = 0 if param.value < .5 else 1

    def _heuristic(self, lo=.2, hi=.8):
        for param in self.parameters():
            if param.value < lo:
                param.value = 0
            elif param.value > hi:
                param.value = 1

    def sat(self):
        def _backup():
            vals = []
            for atom in self.atoms:
                vals.append(atom.truth.value)
            return vals
        vals = _backup()
        def _restore():
            for val, atom in zip(vals, self.atoms):
                atom.truth.value = val
        self._round()
        flag = True
        topo = self._build_topo()
        for x in topo:
            check = x.sat()
            if not check and x in self.symbols:
                flag = False
                break
        if not flag:
            _restore()
        return flag

    def _calc_loss(self) -> Node:
        self.eval()
        loss = 0
        for sym in self.symbols:
            loss = (sym.truth - 1).pow(2) + loss
            
        return loss / len(self.symbols)

    def train(self, lr: float=1.e-3, converge: float=1e-2):
        self.train_mode()
        self.prepare_model(fuzzy=True)
        # only zero_grad once!
        self.zero_grad()
        for iter in range(10000):
            # self.zero_grad()
            loss = self._calc_loss()
            loss.backward()
            for param in self.parameters():
                param.value = clip(param.value - lr * param.grad)
            if iter % 100 == 0:
                print(f'iter: {iter}, loss: {loss.value:.3e}')
            if loss.value < converge:
                print(f"Check converge at iter = {iter}")
                if self.sat():
                    print("Converge!")
                    self._print_atoms()
                    break
                else:
                    lr *= 1.2
                    converge *= .75
                    self._heuristic()
                    print('False sat!')
    @staticmethod
    def from_sat(sat: Sat):
        type_, n_vars, clauses = sat.type_, sat.n_vars, sat.clauses
        in_op, out_op = (OR, AND) if type_ == 'CNF' else (AND, OR)
        kg = KnowledgeBase()
        atoms = []
        for i in range(n_vars):
            atom = Atom(name='var' + str(i + 1))
            atoms.append(atom)
        kg.add(*atoms)
        for i, cl in enumerate(clauses):
            args = []
            for val in cl:
                if val > 0:
                    args.append(atoms[val - 1])
                else:
                    args.append(NOT(atoms[-val - 1]))
            symbol = in_op(*args, name='clause' + str(i + 1))
            kg.add(symbol)

        return kg

# exposed API for creating symbols
def Atom(name: str):
    return Symbol(name=name, op='ASSIGN', children=())

def NOT(symbol, name: str='', value: ValueType=1):
    return Symbol(name=name, op='NOT', children=(symbol,), value=value)

def AND(*args, name: str='', value: ValueType=1):
    return Symbol(name=name, op='AND', children=args, value=value)

def OR(*args, name: str='', value: ValueType=1):
    return Symbol(name=name, op='OR', children=args, value=value)

# cause implies effective
def IMPL(cause_symbol, eff_symbol, name: str='', value: ValueType=1):
    return Symbol(name=name, op='IMPL', children=(cause_symbol, eff_symbol), value=value)

# lhs is equivalent to rhs
def EQUI(lhs, rhs, name: str='', value: ValueType=1):
    return Symbol(name=name, op='EQUI', children=(lhs, rhs), value=value)

# one and only one in args is true
def ONE(*args, name: str='', value: ValueType=1):
    return Symbol(name=name, op='ONE', children=args, value=value)
