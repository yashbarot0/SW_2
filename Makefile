CC = mpicc
CFLAGS = -Wall -O2

poisson: poisson.c
	$(CC) $(CFLAGS) -o poisson poisson.c

poisson2: poisson2.c
	$(CC) $(CFLAGS) -o poisson2 poisson2.c

poisson3: poisson3.c
	$(CC) $(CFLAGS) -o poisson3 poisson3.c
clean:
	rm -f poisson poisson2 poisson3