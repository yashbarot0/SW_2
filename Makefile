CC = mpicc
CFLAGS = -Wall -O2
TARGETS = poisson poisson2 poisson3

# Default target
all: $(TARGETS)

poisson: poisson.c
	$(CC) $(CFLAGS) -o poisson poisson.c

poisson2: poisson2.c
	$(CC) $(CFLAGS) -o poisson2 poisson2.c -lm

poisson3: poisson3.c
	$(CC) $(CFLAGS) -o poisson3 poisson3.c -lm

# Plot generation
heatmap: heatmap.png

heatmap.png: global_grid.txt
	python plot_heatmap.py

# Cleanup
clean:
	rm -f $(TARGETS) *.txt *.png *.out

.PHONY: all clean heatmap