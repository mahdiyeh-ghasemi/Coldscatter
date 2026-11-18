"""Compare free and van der Waals long-range reference matching.

The physical Lennard-Jones potential already contains -C6/R^6.
LongRange specifies only the reference tail used in matching; it is not
added to the physical interaction a second time.
"""
import coldscatter as cs
from coldscatter import units as u

mu = u.reduced_mass(87.0, 87.0)
c6 = 4700.0
potential = cs.LennardJones(c6, c6 * 11.0**6 / 2)
energy = u.from_kelvin(1e-7)
grid = cs.Grid(4.0, 1500.0, 0.02)

free_model = cs.CoupledChannels(mu, [0.0], [[potential]])
vdw_model = cs.CoupledChannels(
    mu, [0.0], [[potential]],
    long_range=cs.LongRange({6: c6}, r_far=5000.0),
)

free_result = free_model.solve(energy, grid)
vdw_result = vdw_model.solve(energy, grid)

print("Free-reference scattering length:", free_result.scattering_length())
print("C6-reference scattering length:  ", vdw_result.scattering_length())
print("C6-reference K matrix:           ", vdw_result.K)
