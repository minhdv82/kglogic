import math

class Node:
    def __init__(self, value, op=None, children=(), requires_grad=False) -> None:
        self._op = op
        self._children = children
        self.value = value
        self.grad = 0
        self.requires_grad = requires_grad
        self._out_grad = lambda : None

    # we might want to manipulate initial value of grad
    def backward(self, grad=None):
        if not self.requires_grad:
            print('This node does not requires grad!')
            return
        nodes = []
        visited = set()
        def _topo_sort(node: Node):
            if node not in visited and node.requires_grad:
                visited.add(node)
                for child in node._children:
                    _topo_sort(child)
                nodes.append(node)
        _topo_sort(self)
        
        self.grad = 1. if grad is None else grad
        for node in reversed(nodes):
            node._out_grad()

    def __repr__(self):
        return f'<Node object: val = {self.value}, grad = {self.grad}, childrend = {self._children}>'

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        res = Node(self.value + other.value, '+', (self, other),
                   requires_grad=self.requires_grad or other.requires_grad)

        if res.requires_grad:
            def _out_grad():
                self.grad += res.grad
                other.grad += res.grad

            res._out_grad = _out_grad

        return res
        
    def __sub__(self, other):
        return self + (other * (-1))

    def __lsub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        res = Node(self.value * other.value, '*', (self, other),
                   requires_grad=self.requires_grad or other.requires_grad)

        if res.requires_grad:
            def _out_grad():
                self.grad += res.grad * other.value
                other.grad += res.grad * self.value

            res._out_grad = _out_grad

        return res
    
    def __truediv__(self, other):
        return self * pow(other, -1)

    def pow(self, n):
        res = Node(self.value**n, 'pow', (self,), requires_grad=self.requires_grad)

        if res.requires_grad:
            def _out_grad():
                self.grad += n * (self.value)**(n - 1.) * res.grad

            res._out_grad = _out_grad

        return res
    
    def sin(self):
        res = Node(math.sin(self.value), 'sin', (self,), requires_grad=self.requires_grad)

        if res.requires_grad:
            def _out_grad():
                self.grad += math.cos(self.value) * res.grad

            res._out_grad = _out_grad

        return res
    
    def cos(self):
        res = Node(math.cos(self.value), 'cos', (self,), requires_grad=self.requires_grad)

        if res.requires_grad:
            def _out_grad():
                self.grad += -math.sin(self.value) * res.grad

            res._out_grad = _out_grad

        return res
    
    def relu(self):
        res = Node((self.value > 0) * self.value, 'relu', (self,), requires_grad=self.requires_grad)

        if res.requires_grad:
            def _out_grad():
                self.grad += (self.value > 0) * res.grad
            
            res._out_grad = _out_grad

        return res

    def log(self):
        res = Node(math.log(self.value), 'log', (self,), requires_grad=self.requires_grad)

        if res.requires_grad:
            def _out_grad():
                self.grad += res.grad / self.value
            
            res._out_grad = _out_grad

        return res

    # kengo fuzzy stuff here
    def ein(self, other):
        other = other if isinstance(other, Node) else Node(other)
        assert (0 <= self.value <= 1 and 0 <= other.value <= 1), f"You got to be kidding me! {self.value, other.value}"
        _ein  = lambda x, y: (x + y) / (1 + x * y)
        _gein = lambda x, y: ((1 - y*y) / (1 + x*y)**2, (1 - x*x) / (1 + x*y)**2)

        res = Node(_ein(self.value, other.value), 'ein', (self, other),
                   requires_grad=self.requires_grad or other.requires_grad)

        if res.requires_grad:
            def _out_grad():
                us_grad, other_grad = _gein(self.value, other.value)
                self.grad += res.grad * us_grad
                other.grad += res.grad * other_grad

            res._out_grad = _out_grad

        return res
    
    # not(x) := 1 - x
    def __not__(self):
        res = self * (-1) + 1
        res._op = 'not'

        return res

    # or(a, b) := a.ein(b) = b.ein(a) -- kengo rule
    # in fuzzy realm, or operator is not associative
    def __or__(self, other):
        res = self.ein(other)
        res._op = 'or'

        return res
    
    # and(a, b) := a + b - or(a, b) -- Kolmogorov rule
    def __and__(self, other):
        res = (self + other) - self.__or__(other)
        res._op = 'and'

        return res
