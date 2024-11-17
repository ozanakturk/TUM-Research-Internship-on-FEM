from mpi4py import MPI
from dolfinx import mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

# Defining the finite element function space
from dolfinx.fem import FunctionSpace
V = FunctionSpace(domain, ("Lagrange", 1))

from dolfinx import fem
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

import numpy
# Creating facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# Creating the Dirichlet boundary condition
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# Defining the trial and test function
import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Defining the source term
from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(-6))