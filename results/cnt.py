from ase.build import nanotube
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.dos import DOSCalculator
from ase.dft.bandgap import bandgap
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import numpy as np
import ase.io
import sys

#function to calculate the size of the nanotube
def P(L = list()):
  xp = []
  yp = []
  for atom in L:
    x_atom = float(atom[0])
    y_atom = float(atom[1])
    xp.append(x_atom)
    yp.append(y_atom)
  x_max = max(xp)
  x_min = min(xp)
  y_max = max(yp)
  y_min = min(yp)
  d_x = x_max - x_min
  d_y = y_max - y_min
  d_a = (d_x + d_y)/2 # A
  d = d_a * 0.1 #nm
  r = d/2 #nm
  circ = (np.pi * d) # nm
  return circ, r, d

#create nanotube
n = int(sys.argv[1])
m = int(sys.argv[2])
pair = '(' + str(n) + ';' + str(m) + ')'
print(f'Nanotube: {pair}')

def create_cnt(n,m,lenght,bond,vacuum): #converge vacuum
  cnt = nanotube(n, m, length=lenght, bond= bond, symbol='C')
  print('Nanotube created')
  cnt.center(vacuum=vacuum, axis=0) #em A
  cnt.center(vacuum=vacuum, axis=1) #em A
  cnt.pbc = [True, True, True]
  return cnt

cnanotube = create_cnt(n,m,1,1.4,4.068)

num_atoms_uc = len(cnanotube.get_atomic_numbers()) # number_atoms/unit_cell

#define calculator for energy calculation
calc = GPAW( mode = PW(ecut = 450 ), #e_cut = 450 eV parece ok
    kpts = (1,1,5), # k = 5 parece ok
    verbose = True,
    occupations = {'name': 'fermi-dirac', 'width': 0.01},
    #occupations = {'name': 'marzari-vanderbilt', 'width': 0.01}, #for metals
    xc = "PBE",
    txt = f'nano_output_p{n}{m}.txt' )

cnanotube.calc = calc

#geometry optimization
uf = ExpCellFilter( cnanotube , mask = [ False, False, True, False, False, False ])
relax = BFGS(uf, maxstep = 0.2)
relax.run(fmax=0.1)

#save the geometry file
ase.io.write(f"POSCAR_{n}{m}.vasp", cnanotube, direct=True)

#calculate the energy
calc.calculate(cnanotube)
ener = cnanotube.get_potential_energy()
e_a_uc = ener/num_atoms_uc #energy per atom per cell
print(f'Energy: {e_a_uc} eV/atom/unit_cell')

#size of the nanotube
print( f' circunference = {P(cnanotube.get_positions())[0]} (nm);  radius = {P(cnanotube.get_positions())[1]} (nm); diameter = {P(cnanotube.get_positions())[2]} (nm)')
"""
#write the gs to a file
calc.write(f'cnt_{n}{m}_gs.gpw', mode='all')

#calculate the BS
nband = int(5*num_atoms_uc)
#nbands_conv = 2*num_atoms_uc + 10
calc = GPAW(f'cnt_{n}{m}_gs.gpw').fixed_density(
    nbands = 2*nband,
    symmetry = 'off',
    kpts={'path': 'GZ', 'npoints': 50},
    #convergence={'bands':nbands_conv},
    txt=f'band_{n}{m}.txt')
    
cnanotube.calc = calc
calc.calculate(cnanotube)

bs = calc.band_structure()

#get the band gap
gap = bandgap(calc)[0]
print(f'Band gap: {gap} eV')

#calculate the DoS
ef = calc.get_fermi_level()

calc = GPAW(f'cnt_{n}{m}_gs.gpw').fixed_density(
    nbands = 2*nband,
    symmetry = 'off',
    kpts=(1,1,100),
    txt=f'dos_{n}{m}.txt')

p = DOSCalculator.from_calculator(calc)
e = np.arange(ef-10,ef+10,0.01)
dos = p.raw_dos(e, width=0.0)
plt.plot(dos, e, color="blue")
plt.xlabel('DoS')
plt.ylabel('Energy (eV)')
plt.axhline(ef, color='black', linestyle='--', linewidth=1)
plt.ylim(ef-10,ef+10)
plt.savefig(f'dos_{n}{m}.png')

#plot both DoS and BS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

#plot BS
bs.plot(ax=ax1, emax = 10.0)
ax1.set_xlabel('Wave Vector')
ax1.set_ylabel('Energy (eV)')
ax1.axhline(ef, color='black', linestyle='--', linewidth=1)
ax1.set_title(f'Band Structure of ({n}; {m})')
ax1.set_ylim(ef-10,ef+10)
ax1.set_position([0.1, 0.1, 0.65, 0.8])

#plot DoS
ax2.plot(dos, e, color="blue")
ax2.set_xlabel('DoS')
ax2.axhline(ef, color='black', linestyle='--', linewidth=1)
ax2.set_ylim(ef-10,ef+10)
ax2.set_position([0.8, 0.1, 0.15, 0.8])
ax2.set_yticklabels([])
plt.savefig(f'band_dos_{n}{m}.png')
"""
#write the data to a file
file = open(f'data_{n}{m}.txt', "w")

file.write(f'Energy:     {e_a_uc}    eV/atom/cell   \n')
file.write(f'Diameter:     {P(cnanotube.get_positions())[2]}    nm    \n')
file.write('Fermi Level:     {ef}    eV    \n')
file.write('Band gap:     {gap}    eV  \n')

file.close()

print(f'Calculations for nanotube {pair} done!')


