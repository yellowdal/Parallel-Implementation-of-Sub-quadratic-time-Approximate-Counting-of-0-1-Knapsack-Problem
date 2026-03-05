// dc_knapsack_final_nsweep_scalingopt.c
// Final sequential grouped + DC + trimming knapsack prototype (scaling-optimized).
// - Tuned thresholds for empirical scaling
// - CSV logging and n-sweep mode
//
// Compile:
//   gcc -O2 -std=c99 dc_knapsack_final_nsweep_scalingopt.c -lfftw3 -lm -o dc_knapsack_final_nsweep_scalingopt
//
// Usage examples:
//   ./dc_knapsack_final_nsweep_scalingopt
//   ./dc_knapsack_final_nsweep_scalingopt 1
//   ./dc_knapsack_final_nsweep_scalingopt sweep csv
//   ./dc_knapsack_final_nsweep_scalingopt nsweep csv
//

#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>

/* ====== Tunable (scaling-optimized) ====== */
#define DEFAULT_EPS 0.05
#define DEFAULT_N 1000
#define DEFAULT_CAPACITY 200
#define DEFAULT_MAX_W 200

/* Lower threshold so we use FFT earlier for large n */
#define ENUMERO_THRESHOLD 50000u
/* Aggressive trimming multiplier for scaling tests (trades accuracy -> speed) */
#define TRIM_CAP_MULT 3.0




double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}



typedef struct { int idx; double val; } SparseEntry;

/* global used for qsort comparator */
static int *g_scaled_ptr_for_sort = NULL;

/* allocate zeroed fftw array */
static double *zalloc_fftw_d(int n) {
    double *a = (double*) fftw_malloc(sizeof(double) * n);
    if (!a) { fprintf(stderr, "fftw_malloc failed (%d)\n", n); exit(1); }
    memset(a, 0, sizeof(double) * n);
    return a;
}

/* trim dense polynomial -> sparse representation */
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

/* FFT convolution using FFTW r2c/c2r */
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

/* Merge two sparse lists: enumeration when small product, else FFT */
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
        if (!dense) { fprintf(stderr,"calloc failed in merge enum\n"); exit(1); }
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

/* DC merge on sparse lists */
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

/* deterministic instance generator */
static void generate_items(int *w, int *p, int n, int max_w, int max_p) {
    srand(42);
    for (int i = 0; i < n; ++i) {
        w[i] = (rand() % max_w) + 1;
        p[i] = (rand() % max_p) + 1;
    }
}

/* comparator for qsort using global scaled pointer */
static int cmp_idx_func(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    int va = g_scaled_ptr_for_sort[ia], vb = g_scaled_ptr_for_sort[ib];
    return (va > vb) - (va < vb);
}

/* grouped solver (single run) */
void grouped_knapsack_trim_rand(int *weights, int *profits, int n, int capacity,
                                double eps, int mode_random,
                                int *enum_merges, int *fft_merges)
{
    int max_w = 0;
    for (int i = 0; i < n; ++i) if (weights[i] > max_w) max_w = weights[i];

    /* adaptive gamma: theoretical gamma_eps but we cap to avoid huge scaled_capacity */
    int cap_limit = 20000; /* keep scaled capacity reasonable for scaling tests */
    double gamma_eps = eps * (double)max_w / (double)(n > 0 ? n : 1);
    double gamma_cap = (double)capacity / (double)cap_limit;
    double gamma = gamma_eps;
    if (gamma_cap > 0 && gamma_cap < gamma) gamma = gamma_cap;
    if (gamma <= 0.0) gamma = 1.0;

    int *scaled = malloc(sizeof(int) * n);
    if (!scaled) { fprintf(stderr,"malloc failed scaled\n"); exit(1); }

    /* randomized rounding optionally */
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)getpid();
    srand(seed);
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

    printf("Scaling: eps=%.4f gamma=%.6f scaled_capacity=%d mode=%s\n",
           eps, gamma, scaled_capacity, mode_random ? "randomized" : "deterministic");

    /* group items by sorted scaled weight into ~sqrt(n) groups */
    int *idx = malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    g_scaled_ptr_for_sort = scaled;
    qsort(idx, n, sizeof(int), cmp_idx_func);

    int G = (int)ceil(sqrt((double)n));
    if (G < 1) G = 1;
    int base_group_size = (n + G - 1) / G;
    printf("Grouping: n=%d groups=%d base_group_size=%d\n", n, G, base_group_size);

    SparseEntry **group_polys = malloc(sizeof(SparseEntry*) * G);
    int *group_lens = calloc(G, sizeof(int));
    if (!group_polys || !group_lens) { fprintf(stderr,"malloc failed groups\n"); exit(1); }

    /* 🟢 Start build phase timer */
    struct timespec build_t0, build_t1;
    clock_gettime(CLOCK_MONOTONIC, &build_t0);

    for (int gi = 0; gi < G; ++gi) {
        int start = gi * base_group_size;
        int end = (start + base_group_size > n) ? n : start + base_group_size;
        int count = end - start;
        if (count <= 0) { group_polys[gi] = NULL; group_lens[gi] = 0; continue; }

        SparseEntry **polys = malloc(sizeof(SparseEntry*) * count);
        int *plens = malloc(sizeof(int) * count);
        for (int j = 0; j < count; ++j) {
            int sw = scaled[idx[start + j]];
            polys[j] = malloc(sizeof(SparseEntry) * 2);
            polys[j][0] = (SparseEntry){0, 1.0};
            polys[j][1] = (SparseEntry){sw, 1.0};
            plens[j] = 2;
        }

        int out_n = 0;
        SparseEntry *merged = dc_merge_sparse(polys, plens, 0, count, scaled_capacity,
                                              eps, &out_n, enum_merges, fft_merges);

        group_polys[gi] = merged;
        group_lens[gi] = out_n;
        printf("Group %d built: size=%d\n", gi, out_n);

        for (int j = 0; j < count; ++j) free(polys[j]);
        free(polys); free(plens);
    }

    /* 🟢 End build phase timer */
    clock_gettime(CLOCK_MONOTONIC, &build_t1);
    double build_time = (build_t1.tv_sec - build_t0.tv_sec) + 
                        (build_t1.tv_nsec - build_t0.tv_nsec) / 1e9;

    /* 🟢 Start merge phase timer */
    struct timespec merge_t0, merge_t1;
    clock_gettime(CLOCK_MONOTONIC, &merge_t0);

    /* final merge of nonempty groups */
    int nonempty = 0;
    for (int gi = 0; gi < G; ++gi) if (group_lens[gi] > 0) nonempty++;
    SparseEntry **to_merge = malloc(sizeof(SparseEntry*) * (nonempty > 0 ? nonempty : 1));
    int *to_lens = malloc(sizeof(int) * (nonempty > 0 ? nonempty : 1));
    int pos = 0;
    for (int gi = 0; gi < G; ++gi) if (group_lens[gi] > 0) {
        to_merge[pos] = group_polys[gi];
        to_lens[pos] = group_lens[gi];
        pos++;
    }

    int final_n = 0;
    SparseEntry *final_sp = NULL;
    if (nonempty > 0)
        final_sp = dc_merge_sparse(to_merge, to_lens, 0, nonempty,
                                   scaled_capacity, eps, &final_n, enum_merges, fft_merges);

    clock_gettime(CLOCK_MONOTONIC, &merge_t1);
    double merge_time = (merge_t1.tv_sec - merge_t0.tv_sec) +
                        (merge_t1.tv_nsec - merge_t0.tv_nsec) / 1e9;

    long double total = 0.0L;
    if (final_sp)
        for (int i = 0; i < final_n; ++i)
            if (final_sp[i].idx <= scaled_capacity) total += final_sp[i].val;

    printf("\n--- Result ---\nNon-empty groups merged: %d\nApproximate (scaled) count = %.0Lf\n",
           nonempty, total);

    printf("[Summary] n=%d cap=%d eps=%.3f time=%.6fs final_n=%d gamma=%.6f mode=%s enum_merges=%d fft_merges=%d\n",
           n, capacity, eps, merge_time, final_n, gamma, mode_random ? "rand" : "det", *enum_merges, *fft_merges);

    /* 🟢 Timing breakdown like MPI */
    printf("\n--- Timing Breakdown ---\n");
    printf("Build phase: %.6f s\n", build_time);
    printf("Final merge phase: %.6f s\n", merge_time);
    printf("Total (build+merge): %.6f s\n", build_time + merge_time);
    printf("------------------------\n");

    /* append CSV */
    FILE *f = fopen("knapsack_results_scaling.csv", "a");
    if (f) {
        fprintf(f, "%d,%d,%.3f,%.6f,%d,%.6f,%s,%d,%d\n",
                n, capacity, eps, merge_time, final_n, gamma,
                mode_random ? "rand" : "det", *enum_merges, *fft_merges);
        fclose(f);
    }

    for (int gi = 0; gi < G; ++gi) if (group_lens[gi] > 0) free(group_polys[gi]);
    free(group_polys); free(group_lens); free(to_merge); free(to_lens);
    if (final_sp) free(final_sp);
    free(idx); free(scaled);
}

/* driver: sweep modes and CSV header */
int main(int argc, char **argv) {
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

    int capacity = DEFAULT_CAPACITY;
    int max_w = DEFAULT_MAX_W;

    if (csv_flag) {
        FILE *f = fopen("knapsack_results_scaling.csv", "w");
        if (f) {
            fprintf(f, "n,cap,eps,time,final_n,gamma,mode,enum_merges,fft_merges\n");
            fclose(f);
        }
    }

   if (!sweep_eps_mode && !n_sweep_mode) {
    int n = DEFAULT_N;
    int *weights = malloc(sizeof(int) * n);
    int *profits = malloc(sizeof(int) * n);
    generate_items(weights, profits, n, max_w, 15);

    printf("Generated %d items, capacity=%d, max_w=%d\n", n, capacity, max_w);
    for (int i = 0; i < 10 && i < n; ++i)
        printf(" item %3d: w=%2d\n", i, weights[i]);

    int enum_cnt = 0, fft_cnt = 0;

    double total_start = now();     // <-- Start total timer
    double build_start = now();     // <-- Start build timer

    // === Run main computation ===
    grouped_knapsack_trim_rand(weights, profits, n, capacity,
                               DEFAULT_EPS, mode_random, &enum_cnt, &fft_cnt);

    double build_end = now();
    double total_end = build_end;   // Right now grouped_knapsack handles both build+merge

    double total_time = total_end - total_start;
    double build_time = build_end - build_start;

    printf("\n--- Timing Summary ---\n");
    printf("Build + Merge phase: %.6f s\n", build_time);
    printf("Total runtime: %.6f s\n", total_time);
    printf("----------------------\n");

    free(weights);
    free(profits);
    return 0;
}


    if (sweep_eps_mode) {
        printf("\n--- Parameter Sweep (eps) Mode ---\nMode = %s\n\n", mode_random ? "Randomized" : "Deterministic");
        double eps_list[] = {0.01, 0.02, 0.05, 0.1};
        int eps_n = sizeof(eps_list) / sizeof(double);
        for (int i = 0; i < eps_n; ++i) {
            double eps = eps_list[i];
            int n = DEFAULT_N;
            int *weights = malloc(sizeof(int) * n);
            int *profits = malloc(sizeof(int) * n);
            generate_items(weights, profits, n, max_w, 15);
            printf("==== Running with eps = %.3f ====\n", eps);
            int enum_cnt = 0, fft_cnt = 0;
            grouped_knapsack_trim_rand(weights, profits, n, capacity, eps, mode_random, &enum_cnt, &fft_cnt);
            printf("\n");
            free(weights); free(profits);
        }
        return 0;
    }

    if (n_sweep_mode) {
        printf("\n--- n-scaling Sweep Mode ---\nMode = %s\n\n", mode_random ? "Randomized" : "Deterministic");
        /* up to 6400 by default; adjust the list if you want bigger */
        int n_list[] = {100, 200, 400, 800, 1600, 3200, 6400};
        int nn = sizeof(n_list) / sizeof(int);
        for (int i = 0; i < nn; ++i) {
            int n = n_list[i];
            int *weights = malloc(sizeof(int) * n);
            int *profits = malloc(sizeof(int) * n);
            generate_items(weights, profits, n, max_w, 15);

            printf(">>> Running n = %d\n", n);
            int enum_cnt = 0, fft_cnt = 0;
            grouped_knapsack_trim_rand(weights, profits, n, capacity, DEFAULT_EPS, mode_random, &enum_cnt, &fft_cnt);
            printf("\n");
            free(weights); free(profits);
        }
        return 0;
    }

    return 0;
}

