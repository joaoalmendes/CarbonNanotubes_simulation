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

def create_cnt(n,m,lenght,bond,vacuum):
  cnt = nanotube(n, m, length=lenght, bond= bond, symbol='C')
  print('Nanotube created')
  cnt.center(vacuum=vacuum, axis=0) #em A
  cnt.center(vacuum=vacuum, axis=1) #em A
  return cnt

def band_dos(n,m, calc,nband, check):
  if check == True:
    calc.write(f'cnt_{n}{m}_gs.gpw', mode='all')
    #dos
    ef = calc.get_fermi_level()
    #dos
    calc = GPAW('WC_gs.gpw').fixeddensity(
      nbands = nband,
      symmetry = 'off',
      kpts=(2,2,10),
      txt=f'dos{n}{m}.txt')

    p = DOSCalculator.from_calculator(calc)
    e = np.arange(ef-2,ef+1.5,0.01)
    dos = p.raw_dos(e, width=0.0)
    #e, dos = calc.get_dos(spin=0, npts=5001, width=None)
    #band_structure
    calc = GPAW(f'cnt_{n}{m}_gs.gpw').fixed_density(
      nbands = nband,
      symmetry = 'off',
      kpts={'path': 'GXG', 'npoints': 150},
      convergence={'bands':10},
      txt=f'band_{n}{m}.txt')
    bs = calc.band_structure()
    
    #plot both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    # Plot band structure
    bs.scatter(ax=ax1)
    ax1.set_xlabel('Wave Vector')
    ax1.set_ylabel('Energy (eV)')
    ax1.axhline(ef, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'Band Structure of ({n}; {m})')
    ax1.set_ylim(ef-2,ef+1.5)
    ax1.set_position([0.1, 0.1, 0.65, 0.8])

    # Plot DoS
    ax2.plot(dos, e, color="blue")
    ax2.set_xlabel('DoS')
    ax2.axhline(ef, color='black', linestyle='--', linewidth=1)
    ax2.set_ylim(ef-2,ef+1.5)
    ax2.set_position([0.8, 0.1, 0.15, 0.8])
    ax2.set_yticklabels([])
    plt.savefig(f'band_dos_{n}{m}_noGO.png')
  else: pass
  return

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

i = 0
for n in range(2,4):
    m =n
  #for m in range(2,3):
    #x = []
    #E = []
    #for k in range(2,10,1):
    #for m in range(5,7):
    #m = n
    pair = '(' + str(n) + ';' + str(m) + ')'
    print(pair)
    cnanotube = create_cnt(n, m, 1, 1.4, 4.068)
    calc = GPAW( mode = PW(ecut =450 ), #e_cut = 450 eV parece ok
        kpts = (1,1,5), # k = 5 parece ok
        verbose = True,
        occupations = {'name': 'fermi-dirac', 'width': 0.05}, #default = 0.05
        xc = "PBE",
        txt = f'nano_output_p{n}{m}.txt' )
    cnanotube.calc = calc
    num_atoms_uc = len(cnanotube.get_atomic_numbers()) # number/atoms/unit_cell

    if num_atoms_uc <= 115: #cnt's muito grandes têm muitas bandas e dá 'kill' à run
      try:
        """
        uf = ExpCellFilter( cnanotube , mask = [ False, False, True, False, False, False ])
        relax = BFGS(uf, maxstep = 0.1 )
        relax.run(fmax=0.01)
        """
        calc.calculate(cnanotube)
        print( f' circunference = {P(cnanotube.get_positions())[0]} (nm);  radius = {P(cnanotube.get_positions())[1]} (nm); diameter = {P(cnanotube.get_positions())[2]} (nm)')
        ener = cnanotube.get_potential_energy()
        
        #x.append(pair)
        #E.append(ener/num_atoms_uc)
        e_a_uc = cnanotube.get_potential_energy()/num_atoms_uc
        print(f'Formation Energy: {e_a_uc} eV/atom/unit_cell')
        #print(cnanotube.get_cell_lengths_and_angles())
        
        #ase.io.write(f"POSCAR_{n}{m}.vasp", cnanotube, direct=True)

        #lat = cnanotube.cell.get_bravais_lattice()
        #print(list(lat.get_special_points()))
        #path = lat.special_path
        nband = int(6.3*num_atoms_uc) + 1
        band_dos(n,m,calc,nband,True)
      except:
          #x.append(pair)
          #E.append(None)
          print(f'Pair {pair} doesnt work')
    else:
      #x.append(pair)
      #E.append(None)
      print(f'Number of atoms too big (>115): pair {pair}')
    #plt.subplot(2,2,1)
    #plt.xlabel("SWCNTs")
    #plt.ylabel(" Total Energy (eV/cell/atom)")
    #plt.title(f"SWCNTs Formation Energy")
    #plt.grid(linewidth = 0.5)
    #plt.scatter(x,E, color=colors[i], label=f'{pair}')
    if i == 6:
      i = 0
    else: i += 1
#plt.legend(loc = 'upper left')
#plt.show()
