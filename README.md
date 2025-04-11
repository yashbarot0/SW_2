# HPC Software II Programming Exercises Report
**Name**: Yashkumar Barot 
---

## 1. 1D Decomposition with MPI_Cart
### Approach
- Implemented MPI Cartesian topology
- Used `MPI_Cart_shift` for neighbor discovery
- Verified with rank neighbor printouts

### Results
$ mpirun -np 4 ./poisson1d
Rank 0 neighbors: left=-2, right=1
Rank 1 neighbors: left=0, right=2
...


2. Poisson Equation Solution
Analytic vs Numerical Solution
Grid Size	Max Error (2000 iterations)
15	1.84e-4
31	3.66e-5
Observation: Finer grids yield better accuracy as expected.

3. I/O and Visualization
Implementation
Added write_grid() and GatherGrid() functions

Generated heatmap using Python:
