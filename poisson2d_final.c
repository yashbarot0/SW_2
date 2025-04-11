#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 2000

// Function to apply boundary conditions
void apply_boundary_conditions(double **grid, int x_start, int y_start, 
                              int nx_local, int ny_local, int nx_global, 
                              int ny_global, double h) {
    // Left boundary (x=0)
    if (x_start == 0) {
        for (int j = 1; j <= ny_local; j++) { // Only interior+ghost rows
            int global_y = y_start + j - 1;
            if (global_y >= 0 && global_y < ny_global) {
                double y = global_y * h;
                grid[j][0] = y / (1.0 + y*y);
            }
        }
    }

    // Right boundary (x=1)
    if (x_start + nx_local == nx_global) {
        for (int j = 1; j <= ny_local; j++) {
            int global_y = y_start + j - 1;
            if (global_y >= 0 && global_y < ny_global) {
                double y = global_y * h;
                grid[j][nx_local+1] = y / (4.0 + y*y);
            }
        }
    }

    // Bottom boundary (y=0)
    if (y_start == 0) {
        for (int i = 0; i < nx_local + 2; i++) {
            grid[0][i] = 0.0;
        }
    }

    // Top boundary (y=1)
    if (y_start + ny_local == ny_global) {
        for (int i = 0; i < nx_local + 2; i++) {
            int global_x = x_start + i - 1;
            if (global_x >= 0 && global_x < nx_global) {
                double x = global_x * h;
                grid[ny_local+1][i] = 1.0 / ((1.0 + x)*(1.0 + x) + 1.0);
            }
        }
    }
}

// Ghost exchange with bounds checking
void exchange_ghosts(double **grid, int nx_local, int ny_local,
                    MPI_Comm comm, int left, int right, int up, int down,
                    MPI_Datatype column_type) {
    MPI_Status status;
    const int TAG = 0;

    // Left/Right exchange (contiguous rows)
    if (left != MPI_PROC_NULL) {
        MPI_Sendrecv(&grid[1][1], ny_local, MPI_DOUBLE, left, TAG,
                    &grid[1][nx_local+1], ny_local, MPI_DOUBLE, right, TAG,
                    comm, &status);
    }
    if (right != MPI_PROC_NULL) {
        MPI_Sendrecv(&grid[1][nx_local], ny_local, MPI_DOUBLE, right, TAG,
                    &grid[1][0], ny_local, MPI_DOUBLE, left, TAG,
                    comm, &status);
    }

    // Up/Down exchange (strided columns)
    if (up != MPI_PROC_NULL) {
        MPI_Sendrecv(&grid[1][1], 1, column_type, up, TAG,
                    &grid[ny_local+1][1], 1, column_type, down, TAG,
                    comm, &status);
    }
    if (down != MPI_PROC_NULL) {
        MPI_Sendrecv(&grid[ny_local][1], 1, column_type, down, TAG,
                    &grid[0][1], 1, column_type, up, TAG,
                    comm, &status);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <grid_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    
    const int n = atoi(argv[1]);
    const double h = 1.0 / (n - 1);

    // Create 2D Cartesian topology
    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm comm_2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);

    // Get processor coordinates and neighbors
    int coords[2];
    MPI_Cart_coords(comm_2d, rank, 2, coords);
    int left, right, up, down;
    MPI_Cart_shift(comm_2d, 0, 1, &left, &right);
    MPI_Cart_shift(comm_2d, 1, 1, &up, &down);

    // Calculate local grid dimensions with remainder handling
    const int base_x = n / dims[0];
    const int rem_x = n % dims[0];
    const int nx_local = coords[0] < rem_x ? base_x + 1 : base_x;
    
    const int base_y = n / dims[1];
    const int rem_y = n % dims[1];
    const int ny_local = coords[1] < rem_y ? base_y + 1 : base_y;

    const int x_start = coords[0] * base_x + ((coords[0] < rem_x) ? coords[0] : rem_x);
    const int y_start = coords[1] * base_y + ((coords[1] < rem_y) ? coords[1] : rem_y);

    // Allocate grid with ghost cells (initialize to 0)
    double **u_old = (double**)calloc((ny_local + 2), sizeof(double*));
    double **u_new = (double**)calloc((ny_local + 2), sizeof(double*));
    for (int j = 0; j < ny_local + 2; j++) {
        u_old[j] = (double*)calloc((nx_local + 2), sizeof(double));
        u_new[j] = (double*)calloc((nx_local + 2), sizeof(double));
    }

    // Create column datatype
    MPI_Datatype column_type;
    MPI_Type_vector(ny_local, 1, nx_local + 2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    apply_boundary_conditions(u_old, x_start, y_start, nx_local, ny_local, n, n, h);

    // Jacobi iterations
    for (int iter = 0; iter < MAX_ITER; iter++) {
        exchange_ghosts(u_old, nx_local, ny_local, comm_2d, left, right, up, down, column_type);

        // Update interior points
        for (int j = 1; j <= ny_local; j++) {
            for (int i = 1; i <= nx_local; i++) {
                u_new[j][i] = 0.25 * (u_old[j-1][i] + u_old[j+1][i] +
                                      u_old[j][i-1] + u_old[j][i+1]);
            }
        }

        // Swap pointers
        double **tmp = u_old;
        u_old = u_new;
        u_new = tmp;

        apply_boundary_conditions(u_old, x_start, y_start, nx_local, ny_local, n, n, h);
    }

    // Cleanup
    MPI_Type_free(&column_type);
    MPI_Comm_free(&comm_2d);
    for (int j = 0; j < ny_local + 2; j++) {
        free(u_old[j]);
        free(u_new[j]);
    }
    free(u_old);
    free(u_new);

    MPI_Finalize();
    return 0;
}