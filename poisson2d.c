#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 2000

void apply_boundary_conditions(double **grid, int x_start, int y_start, 
                              int nx_local, int ny_local, int nx_global, 
                              int ny_global, double h) {
    // Left boundary (x=0)
    if (x_start == 0) {
        for (int j = 0; j < ny_local + 2; j++) {
            int global_y = y_start + j - 1;
            if (global_y < 0 || global_y >= ny_global) continue;
            double y = global_y * h;
            grid[j][0] = y / (1.0 + y*y);
        }
    }

    // Right boundary (x=1)
    if (x_start + nx_local == nx_global) {
        for (int j = 0; j < ny_local + 2; j++) {
            int global_y = y_start + j - 1;
            if (global_y < 0 || global_y >= ny_global) continue;
            double y = global_y * h;
            grid[j][nx_local+1] = y / (4.0 + y*y);
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
            if (global_x < 0 || global_x >= nx_global) continue;
            double x = global_x * h;
            grid[ny_local+1][i] = 1.0 / ((1.0 + x)*(1.0 + x) + 1.0);
        }
    }
}

void exchange_ghosts(double **grid, int nx_local, int ny_local,
                    MPI_Comm comm, int left, int right, int up, int down,
                    MPI_Datatype column_type) {
    MPI_Status status;
    
    // Left/Right exchange (contiguous rows)
    MPI_Sendrecv(&grid[1][1], ny_local, MPI_DOUBLE, left, 0,
                &grid[1][nx_local+1], ny_local, MPI_DOUBLE, right, 0,
                comm, &status);
    MPI_Sendrecv(&grid[1][nx_local], ny_local, MPI_DOUBLE, right, 1,
                &grid[1][0], ny_local, MPI_DOUBLE, left, 1,
                comm, &status);

    // Up/Down exchange (strided columns)
    MPI_Sendrecv(&grid[1][1], 1, column_type, up, 2,
                &grid[ny_local+1][1], 1, column_type, down, 2,
                comm, &status);
    MPI_Sendrecv(&grid[ny_local][1], 1, column_type, down, 3,
                &grid[0][1], 1, column_type, up, 3,
                comm, &status);
}

void GatherGrid2D(double **local_grid, int nx_local, int ny_local, 
                 int nx_global, int ny_global, MPI_Comm comm, 
                 int rank, double **global_grid) {
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    if (rank == 0) {
        // Initialize global grid
        for (int j = 0; j < ny_global; j++) {
            for (int i = 0; i < nx_global; i++) {
                global_grid[j][i] = 0.0;
            }
        }
    }

    // Create temporary buffer for local data (without ghosts)
    double *sendbuf = (double*)malloc(nx_local * ny_local * sizeof(double));
    for (int j = 0; j < ny_local; j++) {
        for (int i = 0; i < nx_local; i++) {
            sendbuf[j*nx_local + i] = local_grid[j+1][i+1];
        }
    }

    // Gather data to rank 0
    int *recvcounts = NULL;
    int *displs = NULL;
    double *recvbuf = NULL;
    
    if (rank == 0) {
        recvcounts = (int*)malloc(nprocs * sizeof(int));
        displs = (int*)malloc(nprocs * sizeof(int));
        
        int cnt = 0;
        for (int p = 0; p < nprocs; p++) {
            int coords_p[2];
            MPI_Cart_coords(comm, p, 2, coords_p);
            int nx_p = (coords_p[0] < (nx_global % dims[0])) ? 
                       (nx_global/dims[0] + 1) : (nx_global/dims[0]);
            int ny_p = (coords_p[1] < (ny_global % dims[1])) ? 
                       (ny_global/dims[1] + 1) : (ny_global/dims[1]);
            
            recvcounts[p] = nx_p * ny_p;
            displs[p] = cnt;
            cnt += recvcounts[p];
        }
        recvbuf = (double*)malloc(cnt * sizeof(double));
    }

    MPI_Gatherv(sendbuf, nx_local * ny_local, MPI_DOUBLE,
               recvbuf, recvcounts, displs, MPI_DOUBLE,
               0, comm);

    if (rank == 0) {
        // Reconstruct global grid
        int idx = 0;
        for (int p = 0; p < nprocs; p++) {
            int coords_p[2];
            MPI_Cart_coords(comm, p, 2, coords_p);
            
            int nx_p = (coords_p[0] < (nx_global % dims[0])) ? 
                      (nx_global/dims[0] + 1) : (nx_global/dims[0]);
            int ny_p = (coords_p[1] < (ny_global % dims[1])) ? 
                      (ny_global/dims[1] + 1) : (ny_global/dims[1]);
            
            int x_start = coords_p[0] * (nx_global/dims[0]) + 
                         ((coords_p[0] < (nx_global % dims[0])) ? coords_p[0] : (nx_global % dims[0]));
            int y_start = coords_p[1] * (ny_global/dims[1]) + 
                         ((coords_p[1] < (ny_global % dims[1])) ? coords_p[1] : (ny_global % dims[1]));

            for (int j = 0; j < ny_p; j++) {
                for (int i = 0; i < nx_p; i++) {
                    int global_x = x_start + i;
                    int global_y = y_start + j;
                    if (global_x < nx_global && global_y < ny_global) {
                        global_grid[global_y][global_x] = recvbuf[idx++];
                    }
                }
            }
        }
    }

    free(sendbuf);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
        free(recvbuf);
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

    // Allocate grid with ghost cells
    double **u_old = (double**)malloc((ny_local + 2) * sizeof(double*));
    double **u_new = (double**)malloc((ny_local + 2) * sizeof(double*));
    for (int j = 0; j < ny_local + 2; j++) {
        u_old[j] = (double*)malloc((nx_local + 2) * sizeof(double));
        u_new[j] = (double*)malloc((nx_local + 2) * sizeof(double));
        for (int i = 0; i < nx_local + 2; i++) {
            u_old[j][i] = 0.0;
            u_new[j][i] = 0.0;
        }
    }

    // Create column datatype for vertical ghost exchange
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

        // Swap grids
        double **tmp = u_old;
        u_old = u_new;
        u_new = tmp;

        apply_boundary_conditions(u_old, x_start, y_start, nx_local, ny_local, n, n, h);
    }

    // Gather results to rank 0
    double **global_solution = NULL;
    if (rank == 0) {
        global_solution = (double**)malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++) {
            global_solution[i] = (double*)malloc(n * sizeof(double));
        }
    }

    GatherGrid2D(u_old, nx_local, ny_local, n, n, comm_2d, rank, global_solution);

    // Output results
    if (rank == 0) {
        FILE *fp = fopen("solution_2d.txt", "w");
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                fprintf(fp, "%.6e ", global_solution[j][i]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
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
    if (rank == 0) {
        for (int i = 0; i < n; i++) free(global_solution[i]);
        free(global_solution);
    }

    MPI_Finalize();
    return 0;
}