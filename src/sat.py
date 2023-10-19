from typing import List

class Sat:
    def __init__(self, type_: str, n: int, n_vars: int, clauses: List[List[int]]) -> None:
        self.type_ = type_
        self.n = n
        self.n_vars = n_vars
        self.clauses = clauses

    @staticmethod
    def parse(file_name: str):
        with open(file_name) as f:
            type_ = f.readline().split()[0]
            lines = f.readline().split()
            n, n_vars, n_clauses = int(lines[0]), int(lines[1]), int(lines[2])
            clauses = []
            for _ in range(n_clauses):
                line = f.readline().split()
                cl = []
                for l in line[:n]:
                    cl.append(int(l))
                clauses.append(cl)

            return Sat(type_=type_, n=n, n_vars=n_vars, clauses=clauses)