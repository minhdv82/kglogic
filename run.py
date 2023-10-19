from src.graph import Node
from src.kglogic import *
from src.sat import Sat

def grad_fun():
    x, y = Node(.9, requires_grad=True), Node(0.2, requires_grad=True)
    foo = y + .5
    z = x.ein(foo)
    zz = z.__and__(1.)
    zz = zz.__neg__() + foo
    zz = (zz - .2).pow(2.)
    zz.backward()
    print(x.grad, y.grad)

def kb_fun():
    # a1 = Atom('Kengo is great')
    # a2 = Atom('Rua cop')
    # p1 = IMPL(a2, a1)
    # p1 = AND(a2, NOT(a1))
    # p2 = NOT(a1)

    # kb = KnowledgeBase()
    # kb.add(a1, a2, p1, p2)
    # kb.train()
    # kb.solve()

    # this stuff is from wikipedia
    a = Atom('a')
    m = Atom('m')
    u = Atom('u')
    n = Atom('n')
    v = Atom('v')
    r = Atom('r')
    x = Atom('x')
    c = Atom('c')
    e = Atom('e')
    s = Atom('s')
    w = Atom('w')
    p = Atom('p')
    q = Atom('q')
    y = Atom('y')
    g = Atom('g')
    z = Atom('z')
    o = Atom('o')
    
    s12 = OR(NOT(a), m, u)
    s13 = OR(a, n, u)
    s14 = OR(NOT(a), r, x)
    s15 = OR(NOT(c), NOT(e), s)
    s16 = OR(c, NOT(m), NOT(w))
    s17 = OR(NOT(c), p, x)
    s18 = OR(a, q, s)
    s19 = OR(e, p, s)
    s20 = OR(NOT(y), e, q)
    s21 = OR(e, r, y)
    s22 = OR(NOT(e), r, z)
    s23 = OR(NOT(g), r, x)
    s24 = OR(g, v, NOT(y))
    s25 = OR(m, NOT(n), u)
    s26 = OR(m, NOT(o), NOT(u))
    s27 = OR(m, o, v)

    s1 = OR(NOT(m), NOT(q), s)
    s2 = OR(NOT(m), NOT(r), NOT(s))
    s3 = OR(m, NOT(u), NOT(v))
    s4 = OR(NOT(c), NOT(e), s)
    s5 = OR(NOT(m), x, NOT(z))
    s6 = OR(NOT(n), r, NOT(y))
    s7 = OR(o, r, NOT(w))
    s8 = OR(NOT(p), q, s)
    s9 = OR(r, NOT(w), NOT(x))
    s10 = OR(r, w, NOT(y))
    s11 = OR(r, w, NOT(z))

    kb = KnowledgeBase()
    kb.add(a, m, u, n, v, r, s, x, y, w, c, e, p, q, o, z, g)
    kb.add(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
                  s12, s13, s14, s15, s16, s17, s18 ,s19, s20, s21, s22, s23, s24, s25, s26, s27)
    
    # import os
    # from os.path import join
    # cur_dir = os.path.dirname(__name__)
    # file_name = join(cur_dir, 'examples')
    # file_name = join(file_name, 'sat.txt')
    # sat = Sat.parse(file_name)
    # kb = KnowledgeBase.from_sat(sat)

    kb.train(lr=1.e-2)
    # kb.solve()

if __name__ == '__main__':
    kb_fun()
    # grad_fun()