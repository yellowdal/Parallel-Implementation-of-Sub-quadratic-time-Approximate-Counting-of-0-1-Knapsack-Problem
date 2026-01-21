Parallel Implementation of Sub-quadratic time
Approximate Counting of 0-1 Knapsack Problem



The 0-1 Knapsack problem serves as a widely studied example
of NP-hard problems, arising in optimality, scheduling, and resource
allocation. While traditional algorithms involving dynamic programming
provide an exact solution, they have a pseudo-polynomial time
complexity of 𝑂(𝑛𝐶), which limits the practicality of the algorithm for
larger instances. More recently, there is a theoretical advancement
showing that the group dynamic programming with convolution via FFT
can approximate the optimal 0-1 Knapsack model in sub-quadratic time.
This paper will discuss the analysis and implementation of the
divide/conquer FFT-based approximation model, specifically: (i) a
sequential implementation as a baseline; (ii) an MPI implementation for
parallelization and (iii) a CUDA based GPU-accelerated implementation
utilizing the cuFFT libraries to achieve the maximum degree of
parallelism when performing convolutions. Each implementation
employs polynomial scaling/limitations, trimming in considerations of
resolution, and a sparse representation to minimize computational
"overhead," which produces closely approximated results to the original
NP-hard Knapsack model. Empirical study shows that there are
significant speedups with respect to the sequential baseline due to using
MPI, while the CUDA implementation shows promise for even greater
speedup based on multi-stream batched FFTs.
