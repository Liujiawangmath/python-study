import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh import IntervalMesh
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from fealpy.functionspace import LagrangeFESpace

class EXPData:
    def domain(self):
        return [0.0,1,0]
    
    @cartesian
    def source(self,p):
        '''
        方程源项
        p:自标量x的数组
        '''
        val = np.zeros(p.shape, dtype=np.float64)
        val = val+2
        return val
    
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
# 计算基函数
space = LagrangeFESpace(mesh, p=1)
qf = mesh.integrator(3) 
bcs, ws = qf.get_quadrature_points_and_weights()
phi = bcs[:, np.newaxis, :]
phi2 = space.basis(bcs)
print(phi.shape)
print(phi2.shape)

# 计算基函数的导数 \phi'
node = mesh.entity('node')
cell = mesh.entity('cell')
NC = mesh.number_of_cells()
v = node[cell[:, 1]] - node[cell[:, 0]]
GD = mesh.geo_dimension()
Dlambda = np.zeros((NC, 2, GD), dtype=mesh.ftype)
h2 = np.sum(v**2, axis=-1)
v /= h2.reshape(-1, 1)
Dlambda[:, 0, :] = -v
Dlambda[:, 1, :] = v
Dlambda = Dlambda[np.newaxis, :]
gphi = np.repeat(Dlambda, 3, axis=0)

# 全局刚度矩阵的组装
cm = mesh.entity_measure('cell')
NN = mesh.number_of_nodes()
A = np.einsum('q, qcid, qcjd, c-> cij', ws, gphi, gphi, cm)
I = np.broadcast_to(cell[:, :, None], shape=A.shape)
J = np.broadcast_to(cell[:, None, :], shape=A.shape)
A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(NN, NN), dtype=np.float64) 

# 全局载荷向量的组装
ps = mesh.bc_to_point(bcs)
print(ps)
fval = np.squeeze(pde.source(ps),axis=2)
print(fval)
cm = mesh.entity_measure('cell')
bb = np.einsum('q, qc,qci, c->ci', ws , fval  ,phi, cm, optimize=True)
NN = mesh.number_of_nodes()
cell = mesh.entity('cell')
b = np.zeros(NN, dtype=np.float64)
np.add.at(b, cell, bb)
print(b)

isBdNode = mesh.ds.boundary_node_flag()
uh = np.zeros(NN, dtype=np.float64) 
uh[isBdNode]  = pde.dirichlet((node[isBdNode])).reshape(-1)

b -= A@uh
b[isBdNode] = uh[isBdNode]

bdIdx = np.zeros(A.shape[0], dtype=np.int_)
bdIdx[isBdNode] = 1
D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
A = D0@A@D0 + D1

uh[:] = spsolve(A, b)
print(uh)

# space = LagrangeFESpace(mesh, p=1) 
# bform = BilinearForm(space)
# bform.add_domain_integrator(DiffusionIntegrator(q=3))
# A = bform.assembly()
# lform = LinearForm(space)
# lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=3))
# F = lform.assembly()
# bc = DirichletBC(space, pde.dirichlet) 
# uh = space.function() 
# A, F = bc.apply(A, F, uh)
# uh[:] = spsolve(A, F)
# print(uh)




    
    

