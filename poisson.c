#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Create 1D Cartesian topology
    int dims[1] = {0};
    int periods[1] = {0};
    int reorder = 0;
    MPI_Comm cart_comm;
    MPI_Dims_create(nprocs, 1, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &cart_comm);

    int my_cart_rank;
    MPI_Comm_rank(cart_comm, &my_cart_rank);

    int left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);

    // Example communication using Cartesian neighbors
    double send_buffer[100], recv_buffer[100];
    MPI_Status status;
    MPI_Sendrecv(send_buffer, 100, MPI_DOUBLE, right, 0,
                 recv_buffer, 100, MPI_DOUBLE, left, 0,
                 cart_comm, &status);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}