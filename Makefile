CC = mpicc
CFLAGS = -Wall -O2

poisson: poisson.c
	$(CC) $(CFLAGS) -o poisson poisson.c

clean:
	rm -f poisson