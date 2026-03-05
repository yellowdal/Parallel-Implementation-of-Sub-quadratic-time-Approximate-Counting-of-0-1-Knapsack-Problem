/* dc_knapsack_mpi_full.c
 *
 * MPI parallel version of grouped + DC + trimming knapsack prototype
 * - Uses MPI for parallel group-building and communication of group polys
 * - Rank 0 performs final DC merge and CSV logging
 * - Prints extensive diagnostics similar to the sequential code
 *
 * Compile:
 *   mpicc -O2 -std=c99 dc_knapsack_mpi_full.c -lfftw3 -lm -o dc_knapsack_mpi_full
 *
 * Run examples:
 *   mpirun -np 4 ./dc_knapsack_mpi_full
 *   mpirun -np 4 ./dc_knapsack_mpi_full 1
 *   mpirun -np 4 ./dc_knapsack_mpi_full sweep csv
 *   mpirun -np 4 ./dc_knapsack_mpi_full nsweep csv
 */

#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <stdint.h>
#include <unistd.h>

#define DEFAULT_EPS 0.05
#define DEFAULT_N 1000
#define DEFAULT_CAPACITY 200
#define DEFAULT_MAX_W 200

/* Tunables copied/adapted from your scaling-optimized sequential version */
#define ENUMERO_THRESHOLD 50000u
#define TRIM_CAP_MULT 3.0

typedef struct { int idx; double val; } SparseEntry;

/* global for qsort comparator */
static int *g_scaled_ptr_for_sort = NULL;

/* utility to allocate zeroed array with FFTW alloc */
static double *zalloc_fftw_d(int n) {
    double *a = (double*) fftw_malloc(sizeof(double) * n);
    if (!a) { fprintf(stderr, "fftw_malloc failed (%d)\n", n); exit(1); }
    memset(a, 0, sizeof(double) * n);
    return a;
}

/* ---------- trimming dense -> sparse ---------- */
static SparseEntry *trim_dense_to_sparse(const double *poly, int len,
                                         double delta, int max_keep, int *out_cnt) {
    if (len <= 0) { *out_cnt = 0; return NULL; }
    SparseEntry *tmp = malloc(sizeof(SparseEntry) * len);
    if (!tmp) { fprintf(stderr, "malloc failed in trim\n"); exit(1); }
    int kept = 0;
    double last = 0.0;
    for (int i = 0; i < len; ++i) {
        double v = poly[i];
        if (v <= 0.0) continue;
        if (kept == 0 || v >= last * (1.0 + delta)) {
            tmp[kept].idx = i;
            tmp[kept].val = v;
            last = v;
            kept++;
        }
        if (max_keep > 0 && kept >= max_keep) break;
    }
    SparseEntry *out = malloc(sizeof(SparseEntry) * kept);
    if (kept > 0 && !out) { fprintf(stderr, "malloc failed in trim out\n"); exit(1); }
    for (int i = 0; i < kept; ++i) out[i] = tmp[i];
    free(tmp);
    *out_cnt = kept;
    return out;
}

/* ---------- FFT convolution (real r2c / c2r) ---------- */
static void convolve_r2r_fftw(const double *a, const double *b, double *out, int N) {
    int Nc = N/2 + 1;
    fftw_complex *A = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nc);
    fftw_complex *B = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nc);
    fftw_complex *C = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nc);
    if (!A || !B || !C) { fprintf(stderr, "fftw_malloc failed in conv\n"); exit(1); }

    fftw_plan pA = fftw_plan_dft_r2c_1d(N, (double*)a, A, FFTW_ESTIMATE);
    fftw_plan pB = fftw_plan_dft_r2c_1d(N, (double*)b, B, FFTW_ESTIMATE);
    fftw_plan pinv = fftw_plan_dft_c2r_1d(N, C, out, FFTW_ESTIMATE);

    fftw_execute(pA);
    fftw_execute(pB);

    for (int i = 0; i < Nc; ++i) {
        double ar = A[i][0], ai = A[i][1];
        double br = B[i][0], bi = B[i][1];
        C[i][0] = ar * br - ai * bi;
        C[i][1] = ar * bi + ai * br;
    }
    fftw_execute(pinv);

    for (int i = 0; i < N; ++i) {
        out[i] /= (double)N;
        if (fabs(out[i]) < 1e-12) out[i] = 0.0;
    }

    fftw_destroy_plan(pA);
    fftw_destroy_plan(pB);
    fftw_destroy_plan(pinv);
    fftw_free(A); fftw_free(B); fftw_free(C);
}

/* ---------- merge two sparse lists: hybrid ---------- */
static SparseEntry *merge_sparse_hybrid(const SparseEntry *A, int nA,
                                        const SparseEntry *B, int nB,
                                        int cap, double eps, int *out_n,
                                        int *enum_cnt, int *fft_cnt) {
    if (nA == 0 || nB == 0) { *out_n = 0; return NULL; }

    double delta = eps / 4.0;
    int max_keep = (int)ceil(TRIM_CAP_MULT / eps);
    if (max_keep < 10) max_keep = 10;

    uint64_t prod = (uint64_t)nA * (uint64_t)nB;
    if (prod <= ENUMERO_THRESHOLD) {
        (*enum_cnt)++;
        int LEN = cap + 1;
        double *dense = calloc(LEN, sizeof(double));
        if (!dense) { fprintf(stderr, "calloc failed in merge enum\n"); exit(1); }
        for (int i = 0; i < nA; ++i) {
            int ia = A[i].idx; double va = A[i].val;
            for (int j = 0; j < nB; ++j) {
                int idx = ia + B[j].idx;
                if (idx <= cap) dense[idx] += va * B[j].val;
            }
        }
        SparseEntry *out = trim_dense_to_sparse(dense, LEN, delta, max_keep, out_n);
        free(dense);
        return out;
    }

    (*fft_cnt)++;
    int needed = cap + 1;
    int N = 1;
    while (N < 2 * needed) N <<= 1;
    if (N < 2) N = 2;
    double *da = zalloc_fftw_d(N);
    double *db = zalloc_fftw_d(N);
    double *dout = zalloc_fftw_d(N);

    for (int i = 0; i < nA; ++i) if (A[i].idx <= cap) da[A[i].idx] += A[i].val;
    for (int j = 0; j < nB; ++j) if (B[j].idx <= cap) db[B[j].idx] += B[j].val;

    convolve_r2r_fftw(da, db, dout, N);
    SparseEntry *out = trim_dense_to_sparse(dout, cap + 1, delta, max_keep, out_n);

    fftw_free(da); fftw_free(db); fftw_free(dout);
    return out;
}

/* ---------- divide-and-conquer merge on sparse lists ---------- */
static SparseEntry *dc_merge_sparse(SparseEntry **polys, int *lens,
                                    int l, int r, int cap, double eps, int *out_n,
                                    int *enum_cnt, int *fft_cnt) {
    if (r - l == 1) {
        int n = lens[l];
        SparseEntry *res = NULL;
        if (n > 0) {
            res = malloc(sizeof(SparseEntry) * n);
            memcpy(res, polys[l], sizeof(SparseEntry) * n);
        }
        *out_n = n;
        return res;
    }
    int m = (l + r) / 2;
    int nL = 0, nR = 0;
    SparseEntry *L = dc_merge_sparse(polys, lens, l, m, cap, eps, &nL, enum_cnt, fft_cnt);
    SparseEntry *R = dc_merge_sparse(polys, lens, m, r, cap, eps, &nR, enum_cnt, fft_cnt);

    SparseEntry *res = merge_sparse_hybrid(L, nL, R, nR, cap, eps, out_n, enum_cnt, fft_cnt);
    if (L) free(L);
    if (R) free(R);
    return res;
}

/* ---------- deterministic random instance generator ---------- */
static void generate_items(int *w, int *p, int n, int max_w, int max_p) {
    /* deterministic seed for reproducibility */
    srand(42);
    for (int i = 0; i < n; ++i) {
        w[i] = (rand() % max_w) + 1;
        p[i] = (rand() % max_p) + 1;
    }
}

/* comparator for qsort */
static int cmp_idx_func(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    int va = g_scaled_ptr_for_sort[ia], vb = g_scaled_ptr_for_sort[ib];
    return (va > vb) - (va < vb);
}

/* ---------- Group-building done by each rank (local) ---------- */
/* Each rank builds only groups gi where (gi % size) == rank */
static void build_groups_local(int rank, int size, int *scaled, int n, int scaled_capacity,
                               SparseEntry ***out_group_polys, int **out_group_lens,
                               int *out_G, int *out_local_enum, int *out_local_fft,
                               int *out_groups_built, int *group_first, int *group_last) {
    int G = (int)ceil(sqrt((double)n));
    if (G < 1) G = 1;
    int base_group_size = (n + G - 1) / G;

    SparseEntry **group_polys = malloc(sizeof(SparseEntry*) * G);
    int *group_lens = calloc(G, sizeof(int));
    if (!group_polys || !group_lens) { fprintf(stderr, "malloc failed groups\n"); exit(1); }

    int local_enum = 0, local_fft = 0;
    int groups_built = 0;
    int first = -1, last = -1;

    /* build only groups assigned to this rank */
    for (int gi = 0; gi < G; ++gi) {
        if ((gi % size) != rank) { group_polys[gi] = NULL; group_lens[gi] = 0; continue; }

        int start = gi * base_group_size;
        int end = (start + base_group_size > n) ? n : start + base_group_size;
        int count = end - start;
        if (count <= 0) { group_polys[gi] = NULL; group_lens[gi] = 0; continue; }

        SparseEntry **polys = malloc(sizeof(SparseEntry*) * count);
        int *plens = malloc(sizeof(int) * count);
        for (int j = 0; j < count; ++j) {
            int sw = scaled[start + j]; /* note: here scaled is per original index ordering, but in this MPI build we will have scaled sorted externally before calling */
            /* To keep semantics matching sequential, we will build trivial (1 + x^sw) polys */
            polys[j] = malloc(sizeof(SparseEntry) * 2);
            polys[j][0] = (SparseEntry){0, 1.0};
            polys[j][1] = (SparseEntry){sw, 1.0};
            plens[j] = 2;
        }

        int out_n = 0;
        SparseEntry *merged = dc_merge_sparse(polys, plens, 0, count, scaled_capacity,
                                              DEFAULT_EPS, &out_n, &local_enum, &local_fft);

        group_polys[gi] = merged;
        group_lens[gi] = out_n;

        for (int j = 0; j < count; ++j) free(polys[j]);
        free(polys); free(plens);

        groups_built++;
        if (first == -1) first = gi;
        last = gi;
    }

    *out_group_polys = group_polys;
    *out_group_lens = group_lens;
    *out_G = G;
    *out_local_enum = local_enum;
    *out_local_fft = local_fft;
    *out_groups_built = groups_built;
    if (group_first) *group_first = first;
    if (group_last) *group_last = last;
}

/* Serialize local non-empty groups: compute total_entries and arrays (lens, idxs, vals) */
static void serialize_groups(SparseEntry **group_polys, int *group_lens, int G,
                             int **out_lens, int *out_gcnt,
                             int **out_idxs, double **out_vals, int *out_total_entries) {
    int gcnt = 0;
    int total_entries = 0;
    for (int gi = 0; gi < G; ++gi) if (group_lens[gi] > 0) { gcnt++; total_entries += group_lens[gi]; }

    int *larr = NULL;
    int *idxs = NULL;
    double *vals = NULL;
    if (gcnt > 0) {
        larr = malloc(sizeof(int) * gcnt);
        idxs = malloc(sizeof(int) * total_entries);
        vals = malloc(sizeof(double) * total_entries);
        int pos = 0; int gi_pos = 0;
        for (int gi = 0; gi < G; ++gi) {
            if (group_lens[gi] <= 0) continue;
            larr[gi_pos] = group_lens[gi];
            for (int k = 0; k < group_lens[gi]; ++k) {
                idxs[pos] = group_polys[gi][k].idx;
                vals[pos] = group_polys[gi][k].val;
                pos++;
            }
            gi_pos++;
        }
    }

    *out_lens = larr;
    *out_gcnt = gcnt;
    *out_idxs = idxs;
    *out_vals = vals;
    *out_total_entries = total_entries;
}

/* Reconstruct groups on root from received buffers */
static SparseEntry **reconstruct_groups_on_root(int G,
                                                int recv_gcnt_total,
                                                int *recv_per_rank_gcnt,
                                                int *recv_lens_flat,       /* concatenation of lens arrays from ranks */
                                                int *recv_idxs_flat,
                                                double *recv_vals_flat,
                                                int *out_group_lens /* size G, initialized to 0 by caller */) {
    /* We'll walk through ranks' contributions and fill groups in increasing gi order.
       BUT the sender ranks sent only the groups they own (gi%size == rank). We assume the root knows G and will place groups in the proper gi slots.
       For simplicity we reconstruct by iterating gi from 0..G-1 and filling from the buffers in the same order ranks sent (which we request in that order).
       Implementation below expects the buffers represent groups in increasing gi for each rank that sent them.
    */
    /* NOTE: The logic that packs and receives in root keeps consistent order, see main recv code. This helper simply consumes the flattened arrays. */
    SparseEntry **group_polys = malloc(sizeof(SparseEntry*) * G);
    for (int i = 0; i < G; ++i) { group_polys[i] = NULL; out_group_lens[i] = 0; }

    int idx_lens_pos = 0;
    int idxs_pos = 0;
    for (int gi = 0; gi < G; ++gi) {
        /* The root doesn't know which gi came from which rank here; root's receive code must place groups in correct gi positions.
           To avoid a complex mapping here, the root receive code (in main) will directly place data into group_polys[gi] for each gi using the per-group lengths and arrays.
           So this function is not used in practice. Kept for completeness. */
        (void)gi;
    }
    return group_polys;
}

/* ---------- main grouped solver in MPI context ---------- */
/* Each rank: build local groups (owner-of-group), serialize and send to root (rank 0).
   Rank 0: receives all groups, reconstructs global group array, does final DC merge and prints+CSV. */
static void grouped_knapsack_mpi(int *weights, int *profits, int n, int capacity,
                                 double eps, int mode_random,
                                 int rank, int size, int csv_flag, int sweep_flag, int nsweep_flag) {
    /* Step 1: compute gamma and scaled weights (done on all ranks; consistent PRNG seed used) */
    int max_w = 0;
    for (int i = 0; i < n; ++i) if (weights[i] > max_w) max_w = weights[i];

    int cap_limit = 20000;
    double gamma_eps = eps * (double)max_w / (double)(n > 0 ? n : 1);
    double gamma_cap = (double)capacity / (double)cap_limit;
    double gamma = gamma_eps;
    if (gamma_cap > 0 && gamma_cap < gamma) gamma = gamma_cap;
    if (gamma <= 0.0) gamma = 1.0;

    int *scaled = malloc(sizeof(int) * n);
    if (!scaled) { fprintf(stderr,"malloc failed scaled\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    /* We will do deterministic scaling across ranks (optionally randomized with same seed) */
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)getpid();
    srand(seed + rank); /* different per rank but we will not rely on that for correctness */
    for (int i = 0; i < n; ++i) {
        double q = (double)weights[i] / gamma;
        int base = (int)floor(q);
        double frac = q - base;
        double r = (double)rand() / (double)RAND_MAX;
        scaled[i] = (mode_random && r < frac) ? base + 1 : base;
        if (scaled[i] < 0) scaled[i] = 0;
    }

    int scaled_capacity = (int)floor((double)capacity / gamma);
    if (scaled_capacity < 0) scaled_capacity = 0;

    /* Print scaling info only on rank 0 for clarity */
    if (rank == 0) {
        printf("Scaling: eps=%.4f gamma=%.6f scaled_capacity=%d mode=%s\n",
               eps, gamma, scaled_capacity, mode_random ? "randomized" : "deterministic");
    }

    /* We will sort items by scaled weight on root and broadcast sorted scaled array to all ranks.
       Sorting on every rank independently could produce different group splits; to keep consistent, root sorts then
       broadcasts the scaled array in the sorted order into a 'scaled_sorted' array that local builders will use with contiguous chunks.
    */
    int *idx = NULL;
    int *scaled_sorted = malloc(sizeof(int) * n);
    if (!scaled_sorted) { fprintf(stderr,"malloc failed scaled_sorted\n"); MPI_Abort(MPI_COMM_WORLD,1); }

    /* Root performs the sort mapping and broadcasts the sorted scaled values */
    if (rank == 0) {
        idx = malloc(sizeof(int) * n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        g_scaled_ptr_for_sort = scaled;
        qsort(idx, n, sizeof(int), cmp_idx_func);
        for (int i = 0; i < n; ++i) scaled_sorted[i] = scaled[idx[i]];
    }

    MPI_Bcast(scaled_sorted, n, MPI_INT, 0, MPI_COMM_WORLD);

    /* Each rank uses scaled_sorted locally and will build groups over that order.
       Now build groups assigned to this rank (round-robin by gi) using scaled_sorted.
       We give build_groups_local a copy of scaled_sorted shifted into contiguous array representing original ordering in groups:
       For simplicity we pass scaled_sorted as-is and in build we assume contiguous layout from start..n.
    */

    /* Local group building */
    double local_start = MPI_Wtime();
    SparseEntry **local_group_polys = NULL;
    int *local_group_lens = NULL;
    int G = 0;
    int local_enum = 0, local_fft = 0;
    int groups_built = 0;
    int group_first = -1, group_last = -1;

    /* We will pass scaled_sorted to local builder but it expects scaled[i] indexable by "start+j".
       That's already satisfied because scaled_sorted is ordered array of size n representing items in sorted order.
    */
    build_groups_local(rank, size, scaled_sorted, n, scaled_capacity,
                       &local_group_polys, &local_group_lens,
                       &G, &local_enum, &local_fft, &groups_built,
                       &group_first, &group_last);
    double local_build_end = MPI_Wtime();
    double local_build_time = local_build_end - local_start;

    /* Print per-rank diagnostics */
    printf("[Rank %d] Built groups: count=%d first=%d last=%d enum=%d fft=%d build_time=%.6f s\n",
           rank, groups_built, group_first, group_last, local_enum, local_fft, local_build_time);

    /* Serialize non-empty groups into lens, idxs, vals arrays */
    int *send_lens = NULL;
    int send_gcnt = 0;
    int *send_idxs = NULL;
    double *send_vals = NULL;
    int send_total_entries = 0;

    serialize_groups(local_group_polys, local_group_lens, G,
                     &send_lens, &send_gcnt, &send_idxs, &send_vals, &send_total_entries);

    /* Each rank sends its send_gcnt and send_total_entries to root (tag 10).
       We'll use blocking sends/recv for simplicity and clarity.
    */
    if (rank == 0) {
        /* Root collects its own data and then receives from others */
        /* Prepare arrays to hold global group slots */
        SparseEntry **global_groups = malloc(sizeof(SparseEntry*) * G);
        int *global_group_lens = calloc(G, sizeof(int));
        if (!global_groups || !global_group_lens) { fprintf(stderr,"malloc failed global groups\n"); MPI_Abort(MPI_COMM_WORLD,1); }

        /* Place root's own groups in global arrays */
        for (int gi = 0; gi < G; ++gi) {
            if (local_group_lens[gi] > 0) {
                int len = local_group_lens[gi];
                global_group_lens[gi] = len;
                global_groups[gi] = malloc(sizeof(SparseEntry) * len);
                for (int k = 0; k < len; ++k) global_groups[gi][k] = local_group_polys[gi][k];
            } else {
                global_group_lens[gi] = 0;
                global_groups[gi] = NULL;
            }
        }

        /* Receive from ranks 1..size-1 */
        for (int src = 1; src < size; ++src) {
            int header[2];
            MPI_Recv(header, 2, MPI_INT, src, 100+src, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int rc_gcnt = header[0];
            int rc_total_entries = header[1];
            if (rc_gcnt == 0) {
                printf("[Root] Received 0 groups from rank %d\n", src);
                continue;
            }
            int *larr = malloc(sizeof(int) * rc_gcnt);
            MPI_Recv(larr, rc_gcnt, MPI_INT, src, 200+src, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int *idxs_flat = malloc(sizeof(int) * rc_total_entries);
            MPI_Recv(idxs_flat, rc_total_entries, MPI_INT, src, 300+src, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double *vals_flat = malloc(sizeof(double) * rc_total_entries);
            MPI_Recv(vals_flat, rc_total_entries, MPI_DOUBLE, src, 400+src, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Now we must place these groups into global_groups in the correct gi positions.
               We assume send order from sender was increasing gi order that sender owns. Thus root can iterate
               over expected gis for that sender: gi such that (gi % size) == src, in ascending gi order,
               and for each such gi, if sender had provided groups (larr entries), assign them sequentially.
            */
            int pos_entries = 0;
            int gi_pos = 0;
            for (int gi = 0; gi < G; ++gi) {
                if ((gi % size) != src) continue;
                if (gi_pos >= rc_gcnt) break; /* sender provided fewer groups than the number of gis it owns (possible if some empty) */
                int len = larr[gi_pos];
                if (len > 0) {
                    global_group_lens[gi] = len;
                    global_groups[gi] = malloc(sizeof(SparseEntry) * len);
                    for (int k = 0; k < len; ++k) {
                        global_groups[gi][k].idx = idxs_flat[pos_entries + k];
                        global_groups[gi][k].val = vals_flat[pos_entries + k];
                    }
                    pos_entries += len;
                } else {
                    global_group_lens[gi] = 0;
                    global_groups[gi] = NULL;
                }
                gi_pos++;
            }

            free(larr); free(idxs_flat); free(vals_flat);
            printf("[Root] Received groups from rank %d: gcnt=%d total_entries=%d\n", src, rc_gcnt, rc_total_entries);
        }

        /* Root: prepare non-empty list for final merge */
        int nonempty = 0;
        for (int gi = 0; gi < G; ++gi) if (global_group_lens[gi] > 0) nonempty++;
        SparseEntry **to_merge = malloc(sizeof(SparseEntry*) * (nonempty > 0 ? nonempty : 1));
        int *to_lens = malloc(sizeof(int) * (nonempty > 0 ? nonempty : 1));
        int pos = 0;
        for (int gi = 0; gi < G; ++gi) if (global_group_lens[gi] > 0) {
            to_merge[pos] = global_groups[gi];
            to_lens[pos] = global_group_lens[gi];
            pos++;
        }

        /* Final merge on root */
        int final_n = 0;
        int enum_cnt_root = 0, fft_cnt_root = 0;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        SparseEntry *final_sp = NULL;
        if (nonempty > 0) final_sp = dc_merge_sparse(to_merge, to_lens, 0, nonempty,
                                                   scaled_capacity, eps, &final_n, &enum_cnt_root, &fft_cnt_root);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        long double total = 0.0L;
        if (final_sp)
            for (int i = 0; i < final_n; ++i)
                if (final_sp[i].idx <= scaled_capacity) total += (long double)final_sp[i].val;

        /* Collect global enum/fft counts by reducing local counts from all ranks */
        int global_enum = enum_cnt_root;
        int global_fft = fft_cnt_root;
        /* root also should include its own local counters (we already included root's contributions in root's dc; but we did not compute total local_enum across ranks) */
        /* For accuracy, we'll perform MPI_Reduce across ranks for enum/fft (local counters sent earlier via MPI_Reduce). We need to collect local_enum and local_fft from all ranks.
           Simpler: do MPI_Reduce now for enum/fft where each rank contributes its local counters. */

        /* But we must have received other ranks' local counts; we will use a reduction across ranks BEFORE final merge in practice.
           For simplicity of this code, do reductions now:
        */
        int local_enum_total = local_enum;
        int local_fft_total = local_fft;
        int sum_enum_all = 0, sum_fft_all = 0;
        MPI_Reduce(&local_enum_total, &sum_enum_all, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_fft_total, &sum_fft_all, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        /* Print results and CSV write */
        printf("\n--- Final Result (Root) ---\nNon-empty groups merged: %d\nApproximate (scaled) count = %.0Lf\nFinal DC merge time = %.6f sec\n",
               nonempty, total, elapsed);

        /* Gather local build times to find max */
        double local_times[size];
        /* Root already has its local build time as local_build_time; gather from others */
        MPI_Gather(&local_build_time, 1, MPI_DOUBLE, local_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double max_local_time = 0.0;
        for (int r = 0; r < size; ++r) if (local_times[r] > max_local_time) max_local_time = local_times[r];

        printf("[Summary] n=%d cap=%d eps=%.3f time=%.6fs final_n=%d gamma=%.6f mode=%s enum_merges=%d fft_merges=%d ranks=%d max_local_build=%.6f\n",
               n, capacity, eps, elapsed, final_n, gamma, mode_random ? "rand" : "det", sum_enum_all, sum_fft_all, size, max_local_time);

        /* CSV logging (rank 0 only) */
        if (csv_flag) {
            FILE *f = fopen("knapsack_results_mpi.csv", "a");
            if (f) {
                fprintf(f, "%d,%d,%.3f,%.6f,%d,%.6f,%s,%d,%d,%d,%.6f\n",
                        n, capacity, eps, elapsed, final_n, gamma,
                        mode_random ? "rand" : "det", sum_enum_all, sum_fft_all, size, max_local_time);
                fclose(f);
            }
        }

        /* cleanup root allocations */
        for (int gi = 0; gi < G; ++gi) if (global_group_lens[gi] > 0) free(global_groups[gi]);
        free(global_groups); free(global_group_lens);
        if (to_merge) free(to_merge);
        if (to_lens) free(to_lens);
        if (final_sp) free(final_sp);
    } else {
        /* Non-root ranks: send header and arrays to root, then participate in reduce/gather for counters/times */

        int header[2];
        header[0] = send_gcnt;
        header[1] = send_total_entries;
        MPI_Send(header, 2, MPI_INT, 0, 100+rank, MPI_COMM_WORLD);
        if (send_gcnt > 0) {
            MPI_Send(send_lens, send_gcnt, MPI_INT, 0, 200+rank, MPI_COMM_WORLD);
            if (send_total_entries > 0) {
                MPI_Send(send_idxs, send_total_entries, MPI_INT, 0, 300+rank, MPI_COMM_WORLD);
                MPI_Send(send_vals, send_total_entries, MPI_DOUBLE, 0, 400+rank, MPI_COMM_WORLD);
            }
        }
        /* Participate in reductions and gather: send local_enum and local_fft via MPI_Reduce */
        MPI_Reduce(&local_enum, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_fft, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        /* participate in gather of local build times */
        MPI_Gather(&local_build_time, 1, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    /* free local memory */
    if (send_lens) free(send_lens);
    if (send_idxs) free(send_idxs);
    if (send_vals) free(send_vals);
    if (local_group_lens) free(local_group_lens);
    if (local_group_polys) free(local_group_polys); /* note: inner small allocations already freed / transferred for root */
    free(scaled_sorted);
    free(scaled);
}

/* ---------- Program driver with MPI-aware CLI ---------- */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

        double total_start = MPI_Wtime();  // Start total timer

    double tick = MPI_Wtick();
    if (rank == 0) {
        printf("MPI parallel knapsack starting (ranks=%d)\nMPI_Wtick() resolution: %.9f sec\n\n", size, tick);
    }

    /* parse CLI on all ranks (we will broadcast flags if needed) */
    int mode_random = 0;
    int sweep_eps_mode = 0;
    int n_sweep_mode = 0;
    int csv_flag = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "1") == 0) mode_random = 1;
        if (strcmp(argv[i], "sweep") == 0) sweep_eps_mode = 1;
        if (strcmp(argv[i], "nsweep") == 0) n_sweep_mode = 1;
        if (strcmp(argv[i], "csv") == 0) csv_flag = 1;
    }

    /* root prepares CSV header if requested */
    if (rank == 0) {
        if (csv_flag) {
            FILE *f = fopen("knapsack_results_mpi.csv", "w");
            if (f) {
                fprintf(f, "n,cap,eps,time,final_n,gamma,mode,enum_merges,fft_merges,ranks,max_local_build\n");
                fclose(f);
            }
        }
    }

    /* barrier to sync ranks before heavy work */
    MPI_Barrier(MPI_COMM_WORLD);

    if (!sweep_eps_mode && !n_sweep_mode) {
        int n = DEFAULT_N;
        int capacity = DEFAULT_CAPACITY;
        int max_w = DEFAULT_MAX_W;

        int *weights = malloc(sizeof(int) * n);
        int *profits = malloc(sizeof(int) * n);
        if (!weights || !profits) { fprintf(stderr,"malloc failed main arrays\n"); MPI_Abort(MPI_COMM_WORLD,1); }

        /* Root generates items then broadcast to all ranks */
        if (rank == 0) generate_items(weights, profits, n, max_w, 15);
        MPI_Bcast(weights, n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(profits, n, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Generated %d items, capacity=%d, max_w=%d\n", n, capacity, max_w);
            for (int i = 0; i < 10 && i < n; ++i) printf(" item %3d: w=%2d\n", i, weights[i]);
            printf("\n");
        }

        grouped_knapsack_mpi(weights, profits, n, capacity, DEFAULT_EPS,
                             mode_random, rank, size, csv_flag, sweep_eps_mode, n_sweep_mode);  
                             
         double total_end = MPI_Wtime();
    double total_time = total_end - total_start;

    if (rank == 0) {
        printf("\n============================\n");
        printf("Overall MPI Runtime Summary\n");
        printf("============================\n");
        printf("Total wall-clock time across ranks: %.6f seconds\n", total_time);
        printf("MPI ranks used: %d\n", size);
        printf("============================\n\n");
    }               

        free(weights); free(profits);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    if (sweep_eps_mode) {
        if (rank == 0) printf("\n--- Parameter Sweep (eps) Mode ---\nMode = %s\n\n", mode_random ? "Randomized" : "Deterministic");
        double eps_list[] = {0.01, 0.02, 0.05, 0.1};
        int eps_n = sizeof(eps_list) / sizeof(double);
        for (int i = 0; i < eps_n; ++i) {
            double eps = eps_list[i];
            int n = DEFAULT_N, capacity = DEFAULT_CAPACITY, max_w = DEFAULT_MAX_W;
            int *weights = malloc(sizeof(int) * n);
            int *profits = malloc(sizeof(int) * n);
            if (rank == 0) generate_items(weights, profits, n, max_w, 15);
            MPI_Bcast(weights, n, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(profits, n, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank == 0) printf("==== Running with eps = %.3f ====\n", eps);
            grouped_knapsack_mpi(weights, profits, n, capacity, eps, mode_random, rank, size, csv_flag, sweep_eps_mode, n_sweep_mode);
            if (rank == 0) printf("\n");
            free(weights); free(profits);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    if (n_sweep_mode) {
        if (rank == 0) printf("\n--- n-scaling Sweep Mode ---\nMode = %s\n\n", mode_random ? "Randomized" : "Deterministic");
        int n_list[] = {100, 200, 400, 800, 1600, 3200, 6400};
        int nn = sizeof(n_list) / sizeof(int);
        int capacity = DEFAULT_CAPACITY, max_w = DEFAULT_MAX_W;
        for (int i = 0; i < nn; ++i) {
            int n = n_list[i];
            int *weights = malloc(sizeof(int) * n);
            int *profits = malloc(sizeof(int) * n);
            if (rank == 0) generate_items(weights, profits, n, max_w, 15);
            MPI_Bcast(weights, n, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(profits, n, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank == 0) printf(">>> Running n = %d\n", n);
            grouped_knapsack_mpi(weights, profits, n, capacity, DEFAULT_EPS, mode_random, rank, size, csv_flag, sweep_eps_mode, n_sweep_mode);
            if (rank == 0) printf("\n");
            free(weights); free(profits);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    double total_end = MPI_Wtime();    // End total timer
    double total_time = total_end - total_start;

    if (rank == 0) {
        printf("\n============================\n");
        printf("Overall MPI Runtime Summary\n");
        printf("============================\n");
        printf("Total wall-clock time across ranks: %.6f seconds\n", total_time);
        printf("MPI ranks used: %d\n", size);
        printf("============================\n\n");
    }


    MPI_Finalize();
    return 0;
}

