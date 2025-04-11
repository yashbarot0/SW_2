#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 2000

// Function to apply boundary conditions (similar to Q2/Q3)
void apply_boundary_conditions(double **grid, int x_start, int y_start, int nx_local, int ny_local, int nx, int ny, double h) {
    // Boundary logic for x=0, x=1, y=0, y=1 (omitted for brevity)
}

// Ghost exchange using MPI_Sendrecv (contiguous and strided)
void exchange_ghosts_sendrecv(double **grid, int nx_local, int ny_local, MPI_Comm comm_2d, 
                              int left, int right, int up, int down, MPI_Datatype column_type) {
    MPI_Status status;
    // Exchange left/right (contiguous rows)
    MPI_Sendrecv(grid[1], nx_local, MPI_DOUBLE, left, 0,
                 grid[ny_local+1], nx_local, MPI_DOUBLE, right, 0, comm_2d, &status);
    MPI_Sendrecv(grid[ny_local], nx_local, MPI_DOUBLE, right, 1,
                 grid[0], nx_local, MPI_DOUBLE, left, 1, comm_2d, &status);

    // Exchange up/down (non-contiguous columns using MPI_Type_vector)
    MPI_Sendrecv(&(grid[0][1]), 1, column_type, up, 2,
                 &(grid[ny_local+1][1]), 1, column_type, down, 2, comm_2d, &status);
    MPI_Sendrecv(&(grid[1][1]), 1, column_type, down, 3,
                 &(grid[0][1]), 1, column_type, up, 3, comm_2d, &status);
}

// Ghost exchange using non-blocking MPI_Isend/Irecv
void exchange_ghosts_nonblocking(double **grid, int nx_local, int ny_local, MPI_Comm comm_2d,
                                 int left, int right, int up, int down, MPI_Datatype column_type) {
    MPI_Request reqs[8];
    // Send/receive left/right
    MPI_Isend(grid[1], nx_local, MPI_DOUBLE, left, 0, comm_2d, &reqs[0]);
    MPI_Irecv(grid[ny_local+1], nx_local, MPI_DOUBLE, right, 0, comm_2d, &reqs[1]);
    MPI_Isend(grid[ny_local], nx_local, MPI_DOUBLE, right, 1, comm_2d, &reqs[2]);
    MPI_Irecv(grid[0], nx_local, MPI_DOUBLE, left, 1, comm_2d, &reqs[3]);

    // Send/receive up/down using column_type
    MPI_Isend(&(grid[0][1]), 1, column_type, up, 2, comm_2d, &reqs[4]);
    MPI_Irecv(&(grid[ny_local+1][1]), 1, column_type, down, 2, comm_2d, &reqs[5]);
    MPI_Isend(&(grid[1][1]), 1, column_type, down, 3, comm_2d, &reqs[6]);
    MPI_Irecv(&(grid[0][1]), 1, column_type, up, 3, comm_2d, &reqs[7]);

    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n = 31; // Grid size
    double h = 1.0 / (n - 1);

    // Create 2D Cartesian grid
    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm comm_2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);

    // Get coordinates and neighbors
    int coords[2];
    MPI_Cart_coords(comm_2d, rank, 2, coords);
    int left, right, up, down;
    MPI_Cart_shift(comm_2d, 0, 1, &left, &right); // X-direction
    MPI_Cart_shift(comm_2d, 1, 1, &up, &down);    // Y-direction

    // Decompose grid
    int nx_local = n / dims[0];
    int ny_local = n / dims[1];
    int x_start = coords[0] * nx_local;
    int y_start = coords[1] * ny_local;

    // Allocate grid with ghost layers
    double **u_old = (double **)malloc((ny_local + 2) * sizeof(double *));
    for (int j = 0; j < ny_local + 2; j++) {
        u_old[j] = (double *)malloc((nx_local + 2) * sizeof(double));
    }

    // Create strided datatype for columns
    MPI_Datatype column_type;
    MPI_Type_vector(ny_local, 1, nx_local + 2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    // Initialize grid and apply BC (similar to Q2/Q3)

    // Jacobi iterations with ghost exchange
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Choose exchange method (comment one)
        exchange_ghosts_sendrecv(u_old, nx_local, ny_local, comm_2d, left, right, up, down, column_type);
        // exchange_ghosts_nonblocking(u_old, nx_local, ny_local, comm_2d, left, right, up, down, column_type);

        // Update interior points (similar to Q2/Q3)
    }

    // Gather global grid onto rank 0 (GatherGrid2D)
    // Error calculation and output (similar to Q2/Q3)

    MPI_Finalize();
    return 0;
}