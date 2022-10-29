import numpy as np
from tqdm import tqdm

class Solver:
    
    def __init__(self, eta, r, show_progress=False):
        self.eta = eta
        self.r = r
        self.show_progress = show_progress
        
    def solve(self):
        raise NotImplementedError()
        
    @staticmethod
    def sample_sphere(d):
        u = np.random.randn(d)
        u /= np.linalg.norm(u)
        return u

class VanillaSZOSolver(Solver):
    
    def __init__(self, eta, r, show_progress=False):
        super().__init__(eta, r, show_progress)
        
    def solve(self, f, x0, iterations):
        fvals = [f(x0)]
        d = len(x0)
        eta = self.eta
        r = self.r

        x = x0
        if self.show_progress:
            for _ in tqdm(range(iterations)):
                u = Solver.sample_sphere(d)
                x = x - eta * d / r * f(x + r * u) * u
                fvals.append(f(x))
        else:
            for _ in range(iterations):
                u = Solver.sample_sphere(d)
                x = x - eta * d / r * f(x + r * u) * u
                fvals.append(f(x))

        return np.array(fvals), x
    
class TPSZOSolver(Solver):
    
    def __init__(self, eta, r, show_progress=False):
        super().__init__(eta, r, show_progress)
        
    def solve(self, f, x0, iterations):
        fvals = [f(x0)]
        d = len(x0)
        eta = self.eta
        r = self.r

        x = x0
        if self.show_progress:
            for _ in tqdm(range(iterations)):
                u = Solver.sample_sphere(d)
                x = x - eta * d / (2 * r) * (f(x + r * u) - f(x - r * u)) * u
                fvals.append(f(x))
        else:
            for _ in range(iterations):
                u = Solver.sample_sphere(d)
                x = x - eta * d / (2 * r) * (f(x + r * u) - f(x - r * u)) * u
                fvals.append(f(x))

        return np.array(fvals), x

class LFSZOSolver(Solver):
    
    def __init__(self, eta, r, alpha=0.9, show_progress=False):
        super().__init__(eta, r, show_progress)
        self.alpha = alpha
        
    def solve(self, f, x0, iterations):
        fvals = [f(x0)]
        d = len(x0)
        eta = self.eta
        r = self.r
        alpha = self.alpha

        xprev = x0
        x = xprev

        if self.show_progress:
            for _ in tqdm(range(iterations)):
                u = Solver.sample_sphere(d)
                xnext = x - eta * d / r * f(x + r * u) * u + alpha * (x - xprev)
                fvals.append(f(xnext))

                xprev = x
                x = xnext
        else:
            for _ in range(iterations):
                u = Solver.sample_sphere(d)
                xnext = x - eta * d / r * f(x + r * u) * u + alpha * (x - xprev)
                fvals.append(f(xnext))

                xprev = x
                x = xnext

        return np.array(fvals), x

class HFSZOSolver(Solver):
    
    def __init__(self, eta, r, beta=1.0, show_progress=False):
        super().__init__(eta, r, show_progress)
        self.beta = beta
        
    def solve(self, f, x0, iterations):
        fvals = [f(x0)]
        d = len(x0)
        eta = self.eta
        r = self.r
        beta = self.beta

        x = x0

        zprev = f(x + r * Solver.sample_sphere(d))
        fprev = zprev

        if self.show_progress:
            for _ in tqdm(range(iterations)):
                u = Solver.sample_sphere(d)
                fcurr = f(x + r * u)
                z = (1-beta) * zprev + fcurr - fprev
                xnext = x - eta * (d / r) * z * u
                fvals.append(f(xnext))

                zprev = z
                fprev = fcurr
                x = xnext
        else:
            for _ in range(iterations):
                u = Solver.sample_sphere(d)
                fcurr = f(x + r * u)
                z = (1-beta) * zprev + fcurr - fprev #(1-beta)*z_old + fun(x_old + r*uk) - fun(x_veryold + r*u_old)
                xnext = x - eta * (d / r) * z * u
                fvals.append(f(xnext))

                zprev = z
                fprev = fcurr
                x = xnext

        return np.array(fvals), x

class HLFSZOSolver(Solver):
    
    def __init__(self, eta, r, alpha=0.9, beta=1.0, show_progress=False):
        super().__init__(eta, r, show_progress)
        self.alpha = alpha
        self.beta = beta
        
    def solve(self, f, x0, iterations):
        fvals = [f(x0)]
        d = len(x0)
        eta = self.eta
        r = self.r
        alpha = self.alpha
        beta = self.beta

        xprev = x0
        x = xprev

        zprev = f(x + r * Solver.sample_sphere(d))
        fprev = zprev
        
        if self.show_progress:
            for _ in tqdm(range(iterations)):
                u = Solver.sample_sphere(d)
                fcurr = f(x + r * u)
                z = (1-beta) * zprev + fcurr - fprev
                xnext = x - eta * d / r * z * u + alpha * (x - xprev)
                fvals.append(f(xnext))

                zprev = z
                fprev = fcurr
                xprev = x
                x = xnext
        else:
            for _ in range(iterations):
                u = Solver.sample_sphere(d)
                fcurr = f(x + r * u)
                z = (1-beta) * zprev + fcurr - fprev
                xnext = x - eta * d / r * z * u + alpha * (x - xprev)
                fvals.append(f(xnext))

                zprev = z
                fprev = fcurr
                xprev = x
                x = xnext
                
        return np.array(fvals), x