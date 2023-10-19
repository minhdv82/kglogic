from .graph import Node

def _and(*nodes) -> Node:
    res = nodes[0]
    for node in nodes[1:]:
        res = res.__and__(node)
    return res

def _or(*nodes) -> Node:
    res = nodes[0]
    for node in nodes[1:]:
        res = res.__or__(node)
    return res

def _not(node: Node) -> Node:
    return node.__not__()

def _impl(lhs: Node, rhs: Node) -> Node:
    return lhs.__not__().__or__(rhs)

def _equi(lhs: Node, rhs: Node) -> Node:
    return _impl(lhs, rhs).__and__(_impl(rhs, lhs))

def _one(*nodes) -> Node:
    if len(nodes) == 1:
        return nodes[0]
    nodes = list(nodes)
    def _get(i):
        return nodes[i].__and__(_or(*(nodes[:i] + nodes[i + 1:])).__not__())
    res = _get(0)
    for i in range(1, len(nodes)):
        res = res.__or__(_get(i))
    return res

def _notd(arg):
    return 1 - arg > .9

def _andd(*args):
    res = 1
    for arg in args:
        res *= arg
    return res > .9

def _ord(*args):
    res = 0
    for arg in args:
        res += arg
    return res > .9

def _impld(lhs, rhs):
    return _ord(_notd(lhs), rhs)

def _equid(lhs, rhs):
    return _andd(_impld(lhs, rhs), _impld(rhs, lhs))

def _oned(*args):
    res = 0
    for arg in args:
        res += arg
    return .9 < res < 1.1

# dict that maps user-defined ops to backend ops
ops_dict_ = {'NOT': _not, 'AND': _and, 'OR': _or, 'IMPL': _impl, 'EQUI': _equi, 'ONE': _one}
# dict for conventional logic ops
opsd_dict_ = {'NOT': _notd, 'AND': _andd, 'OR': _ord, 'IMPL': _impld, 'EQUI': _equid, 'ONE': _oned}
