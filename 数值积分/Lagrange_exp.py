import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh import IntervalMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import LinearForm
from scipy.sparse.linalg import spsolve
from fealpy.fem import DirichletBC
import matplotlib.pyplot as plt


class EXPData:
    def domain(self):
        return [0.0,1,0]
    
    @cartesian
    def source(self,p):
        '''
        方程源项
        p:自标量x的数组
        '''
        return 2
    
    @cartesian
    def dirichlet(self,p)->np.ndarray:
        '''
        dirichlet边界
        '''
        p[0]=0
        p[-1]=0
        return p
    
pde = EXPData()
nx  = 10
domain = pde.domain() 

mesh = IntervalMesh.from_interval(domain, nx=nx)

space = LagrangeFESpace(mesh, p=1) 
bform = BilinearForm(space)
bform.add_domain_integrator(DiffusionIntegrator(q=3))
A = bform.assembly()
lform = LinearForm(space)
lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=3))
F = lform.assembly()
bc = DirichletBC(space, pde.dirichlet) 
uh = space.function() 
A, F = bc.apply(A, F, uh)
print(F)
uh[:] = spsolve(A, F)
print(uh)
