from ase.build import nanotube
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.dos import DOSCalculator
from ase.dft.bandgap import bandgap
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import numpy as np

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

def create_cnt(n,m,lenght,bond,vacuum): #converge vacuum
  cnt = nanotube(n, m, length=lenght, bond= bond, symbol='C')
  positions = cnt.get_positions()
  cnt.center(vacuum=vacuum, axis=0) #em A
  cnt.center(vacuum=vacuum, axis=1) #em A
  cnt.pbc = [True, True, True]
  return cnt, positions

def gs_calculation(n, m, go, converge_parameter, what):
    if what == 'vacuum':
        cnanotube = create_cnt(n,m,1,1.4,converge_parameter)[0]
        positions = create_cnt(n,m,1,1.4,converge_parameter)[1]
    else:
        cnanotube = create_cnt(n,m,1,1.4,4.068)[0]
        positions = create_cnt(n,m,1,1.4,4.068)[1]
    num_atoms_uc = len(cnanotube.get_atomic_numbers()) # number_atoms/unit_cell

    #define calculator for energy calculation
    if what == 'ecut':
        calc = GPAW( mode = PW(ecut = converge_parameter ), #e_cut = 450 eV parece ok
            kpts = (1,1,5), # k = 5 parece ok
            verbose = True,
            occupations = {'name': 'fermi-dirac', 'width': 0.05}, #default = 0.05
            xc = "PBE",
            txt = f'nano_output_p{n}{m}.txt' )
    if what == 'k':
        calc = GPAW( mode = PW(ecut = 450 ), #e_cut = 450 eV parece ok
            kpts = (1,1,converge_parameter), # k = 5 parece ok
            verbose = True,
            occupations = {'name': 'fermi-dirac', 'width': 0.05}, #default = 0.05
            xc = "PBE",
            txt = f'nano_output_p{n}{m}.txt' )
    if what == 'vacuum':
        calc = GPAW( mode = PW(ecut = 450 ), #e_cut = 450 eV parece ok
            kpts = (1,1,5), # k = 5 parece ok
            verbose = True,
            occupations = {'name': 'fermi-dirac', 'width': 0.05}, #default = 0.05
            xc = "PBE",
            txt = f'nano_output_p{n}{m}.txt' )
    
    cnanotube.calc = calc

    if go == True:
        #geometry optimization
        uf = ExpCellFilter( cnanotube , mask = [ False, False, True, False, False, False ])
        relax = BFGS(uf, maxstep = 0.2)
        relax.run(fmax=0.01)

    #calculate the energy
    calc.calculate(cnanotube)
    ener = cnanotube.get_potential_energy()
    e_a_uc = ener/num_atoms_uc #energy per atom per cell
    return e_a_uc, positions

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
i = 0
for n in [2, 4, 6]:
   x = []
   y = []
   new = 0
   m = n
   pair = '(' + str(n) + ';' + str(m) + ')'
   print(f'Doing ({n},{m})')
   for converge_parameter in np.arange(200, 750, 50):
        x.append(converge_parameter)
        try:
            before = abs(gs_calculation(n, m, False, converge_parameter, 'ecut')[0])
            y.append(abs(new - before))
            new = before
        except Exception as e:
            print(e)
            y.append(None)
   plt.xlabel("Cutoff Energy (eV)")
   plt.ylabel("Absolute error for the energy (eV/cell/atom)")
   #plt.title(f"SWCNTs Formation Energy")
   plt.grid(linewidth = 0.5)
   dia = P(gs_calculation(n, m, False, converge_parameter, 'ecut')[1])[2]
   #plt.scatter(x,y, color=colors[i], label=f'{pair}, d={round(dia, 3)} nm')
   plt.semilogy(x, y, marker='o', linestyle='-', color=colors[i], label=f'{pair}, d={round(dia, 3)} nm')
   i += 1
plt.yscale("log")
plt.legend(loc = 0)
plt.savefig(f'converge_ecut_log.pdf', format="pdf", bbox_inches="tight")
#plt.show()