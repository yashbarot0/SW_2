#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 2000

void apply_boundary_conditions(double **grid, int y_start, int y_end, int nx, int local_ny, double h) {
    for (int j = 1; j <= local_ny; j++) {
        int global_y = y_start + (j - 1);
        double y = global_y * h;
        // x=0 boundary
        grid[j][0] = y / (1.0 + y*y);
        // x=1 boundary
        grid[j][nx-1] = y / (4.0 + y*y);
    }

    // y=0 boundary (applied to first real row)
    if (y_start == 0) {
        for (int i = 0; i < nx; i++) {
            grid[1][i] = 0.0;
        }
    }

    // y=1 boundary (applied to last real row)
    if (y_end == (int)(1.0/h)) { // Check if process contains y=1
        for (int i = 0; i < nx; i++) {
            double x = i * h;
            grid[local_ny][i] = 1.0 / ((1.0 + x)*(1.0 + x) + 1.0);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <grid_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int n = atoi(argv[1]);
    if (n < 2) {
        if (rank == 0) printf("Grid size must be at least 2\n");
        MPI_Finalize();
        return 1;
    }

    double h = 1.0 / (n - 1);
    int nx = n, ny = n;

    // Create 1D Cartesian topology
    MPI_Comm cart_comm;
    int dims[1] = {0}, periods[1] = {0};
    MPI_Dims_create(nprocs, 1, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);

    // Get local decomposition
    int coords[1];
    MPI_Cart_coords(cart_comm, rank, 1, coords);
    int quotient = ny / dims[0];
    int remainder = ny % dims[0];
    int local_ny = coords[0] < remainder ? quotient + 1 : quotient;
    int y_start = coords[0] * quotient + (coords[0] < remainder ? coords[0] : remainder);
    int y_end = y_start + local_ny - 1;

    // Allocate memory with ghost rows
    double **u_old = (double **)malloc((local_ny + 2) * sizeof(double *));
    double **u_new = (double **)malloc((local_ny + 2) * sizeof(double *));
    for (int j = 0; j < local_ny + 2; j++) {
        u_old[j] = (double *)malloc(nx * sizeof(double));
        u_new[j] = (double *)malloc(nx * sizeof(double));
        for (int i = 0; i < nx; i++) {
            u_old[j][i] = 0.0;
            u_new[j][i] = 0.0;
        }
    }

    apply_boundary_conditions(u_old, y_start, y_end, nx, local_ny, h);

    int left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);

    // Jacobi iteration
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Exchange ghost cells
        MPI_Sendrecv(u_old[1], nx, MPI_DOUBLE, left, 0,
                     u_old[local_ny + 1], nx, MPI_DOUBLE, right, 0,
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u_old[local_ny], nx, MPI_DOUBLE, right, 1,
                     u_old[0], nx, MPI_DOUBLE, left, 1,
                     cart_comm, MPI_STATUS_IGNORE);

        // Update interior points
        for (int j = 1; j <= local_ny; j++) {
            int global_y = y_start + (j - 1);
            for (int i = 1; i < nx - 1; i++) {
                if (global_y == 0 || global_y == ny - 1) continue;
                u_new[j][i] = 0.25 * (u_old[j-1][i] + u_old[j+1][i] + u_old[j][i-1] + u_old[j][i+1]);
            }
        }

        // Apply BC again to ensure boundaries are maintained
        apply_boundary_conditions(u_new, y_start, y_end, nx, local_ny, h);

        // Swap grids
        double **temp = u_old;
        u_old = u_new;
        u_new = temp;
    }

    // Compute maximum error
    double max_error = 0.0;
    for (int j = 1; j <= local_ny; j++) {
        int global_y = y_start + (j - 1);
        double y = global_y * h;
        for (int i = 0; i < nx; i++) {
            double x = i * h;
            double analytic = y / ((1.0 + x)*(1.0 + x) + y*y);
            double error = fabs(u_old[j][i] - analytic);
            if (error > max_error) max_error = error;
        }
    }

    double global_max_error;
    MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (rank == 0) {
        printf("Grid size: %d, Max error after %d iterations: %.6e\n", n, MAX_ITER, global_max_error);
    }

    // Cleanup
    for (int j = 0; j < local_ny + 2; j++) {
        free(u_old[j]);
        free(u_new[j]);
    }
    free(u_old);
    free(u_new);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}