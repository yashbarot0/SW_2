# HPC Software II Programming Exercises Report
---

## 1. 1D Decomposition with MPI_Cart
### Approach
- Implemented MPI Cartesian topology
- Used `MPI_Cart_shift` for neighbor discovery
- Verified with rank neighbor printouts

## 2. Poisson Equation Solution
Finer grids yield better accuracy as expected.

## 3. I/O and Visualization
- Added write_grid() and GatherGrid() functions. 
- Generated heatmap using Python.



### **Usage**
1. Save code files as:
   - `poisson.c` (Q1)
   - `poisson2.c` (Q2)
   - `poisson3.c` (Q3)
   - `plot_heatmap.py` (visualization)
   - `Makefile`
   - `heatmap.png`

2. Compile and run:
```bash
make              # Compile all programs
make poisson2    # Compile specific question
mpirun -np 4 ./poisson2 31
make heatmap      # Generate visualization
$ mpirun -np 4 ./poisson1d
Output
Rank 0 (World) | Cartesian Rank 0: Left Neighbor = -1, Right Neighbor = 1
Rank 1 (World) | Cartesian Rank 1: Left Neighbor = 0, Right Neighbor = 2
Rank 2 (World) | Cartesian Rank 2: Left Neighbor = 1, Right Neighbor = 3
Rank 3 (World) | Cartesian Rank 3: Left Neighbor = 2, Right Neighbor = -1

Analytic vs Numerical Solution
Grid Size	Max Error (2000 iterations)
[baroty@seagull01 SW_2]$ mpirun -np 4 ./poisson2 15
Grid size: 15, Max error after 2000 iterations: 1.845529e-04
[baroty@seagull01 SW_2]$ mpirun -np 4 ./poisson2 31
Grid size: 31, Max error after 2000 iterations: 3.665882e-05

Observation: Finer grids yield better accuracy as expected.
