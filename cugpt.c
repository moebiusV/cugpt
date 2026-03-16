/*
 * cugpt.c — Optimized C translation of Karpathy's microgpt.py
 *
 * Manual forward/backward with OpenBLAS-accelerated linear algebra.
 * No autograd tape. Analytical gradients computed by hand for each op.
 * Every matrix-vector product goes through BLAS.
 *
 * Build:  cc -O2 -march=native -o cugpt cugpt.c -lopenblas -lm
 * Run:    ./cugpt
 *
 * Architecture: GPT-2 variant (RMSNorm, ReLU, no biases)
 *   Default: vocab=27, n_embd=16, n_head=4, block_size=16, 1 layer
 *   4,192 parameters, character-level tokenizer on names
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <cblas.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Configuration ─────────────────────────────────────────────── */

#define N_LAYER    1
#define N_EMBD     16
#define BLOCK_SIZE 16
#define N_HEAD     4
#define HEAD_DIM   (N_EMBD / N_HEAD)
#define MLP_DIM    (4 * N_EMBD)
#define MAX_VOCAB  64
#define NUM_STEPS  1000
#define LR         0.01
#define BETA1      0.85
#define BETA2      0.99
#define EPS_ADAM   1e-8
#define TEMPERATURE 0.5
#define MAX_SEQ    (BLOCK_SIZE + 2)

/* ── PRNG (xoshiro256**) ──────────────────────────────────────── */

static uint64_t rng_s[4];

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
    uint64_t r = rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = rotl(rng_s[3], 45);
    return r;
}

static void rng_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}

static inline double rng_uniform(void) {
    return (rng_next() >> 11) * 0x1.0p-53;
}

static double rng_gauss(double mean, double std) {
    double u1 = rng_uniform(), u2 = rng_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void shuffle_int(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_uniform() * (i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ── Dataset ───────────────────────────────────────────────────── */

#define MAX_DOCS   40000
#define MAX_DOC_LEN 20

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int  doc_order[MAX_DOCS];
static int  num_docs;
static char uchars[128];
static int  vocab_size;
static int  BOS;

static int char_to_id(char c) {
    for (int i = 0; i < BOS; i++)
        if (uchars[i] == c) return i;
    return -1;
}

static void load_dataset(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    char line[256];
    int char_seen[256] = {0};
    num_docs = 0;
    while (fgets(line, sizeof(line), f)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;
        if (num_docs >= MAX_DOCS) break;
        strncpy(docs[num_docs], line, MAX_DOC_LEN - 1);
        docs[num_docs][MAX_DOC_LEN - 1] = 0;
        for (int i = 0; i < len && i < MAX_DOC_LEN - 1; i++)
            char_seen[(unsigned char)line[i]] = 1;
        num_docs++;
    }
    fclose(f);
    int nc = 0;
    for (int c = 0; c < 256; c++)
        if (char_seen[c]) uchars[nc++] = (char)c;
    BOS = nc;
    vocab_size = nc + 1;
    for (int i = 0; i < num_docs; i++) doc_order[i] = i;
    shuffle_int(doc_order, num_docs);
    printf("num docs: %d\n", num_docs);
    printf("vocab size: %d\n", vocab_size);
}

/* ── Weight Storage ────────────────────────────────────────────── */
/*
 * All weights stored as flat row-major arrays.
 * W[nout][nin] stored contiguously: W[r*nin + c]
 * Gradient arrays mirror the weight layout exactly.
 */

/* Weight matrices (row-major) */
static double W_wte    [MAX_VOCAB * N_EMBD];
static double W_wpe    [BLOCK_SIZE * N_EMBD];
static double W_lm_head[MAX_VOCAB * N_EMBD];
static double W_attn_wq[N_LAYER][N_EMBD * N_EMBD];
static double W_attn_wk[N_LAYER][N_EMBD * N_EMBD];
static double W_attn_wv[N_LAYER][N_EMBD * N_EMBD];
static double W_attn_wo[N_LAYER][N_EMBD * N_EMBD];
static double W_mlp_fc1[N_LAYER][MLP_DIM * N_EMBD];
static double W_mlp_fc2[N_LAYER][N_EMBD * MLP_DIM];

/* Gradient matrices (same layout) */
static double G_wte    [MAX_VOCAB * N_EMBD];
static double G_wpe    [BLOCK_SIZE * N_EMBD];
static double G_lm_head[MAX_VOCAB * N_EMBD];
static double G_attn_wq[N_LAYER][N_EMBD * N_EMBD];
static double G_attn_wk[N_LAYER][N_EMBD * N_EMBD];
static double G_attn_wv[N_LAYER][N_EMBD * N_EMBD];
static double G_attn_wo[N_LAYER][N_EMBD * N_EMBD];
static double G_mlp_fc1[N_LAYER][MLP_DIM * N_EMBD];
static double G_mlp_fc2[N_LAYER][N_EMBD * MLP_DIM];

/* Adam buffers: flatten all params into a single indexing scheme */
#define MAX_PARAMS 8192
static double *all_W[MAX_PARAMS]; /* pointers to each scalar param */
static double *all_G[MAX_PARAMS]; /* pointers to each scalar gradient */
static double adam_m[MAX_PARAMS];
static double adam_v[MAX_PARAMS];
static int    num_params;

static void register_matrix(double *W, double *G, int size) {
    for (int i = 0; i < size; i++) {
        all_W[num_params] = &W[i];
        all_G[num_params] = &G[i];
        num_params++;
    }
}

static void init_matrix(double *W, int size, double std) {
    for (int i = 0; i < size; i++)
        W[i] = rng_gauss(0.0, std);
}

static void init_params(void) {
    num_params = 0;

    init_matrix(W_wte,     vocab_size * N_EMBD, 0.08);
    init_matrix(W_wpe,     BLOCK_SIZE * N_EMBD, 0.08);
    init_matrix(W_lm_head, vocab_size * N_EMBD, 0.08);
    register_matrix(W_wte,     G_wte,     vocab_size * N_EMBD);
    register_matrix(W_wpe,     G_wpe,     BLOCK_SIZE * N_EMBD);
    register_matrix(W_lm_head, G_lm_head, vocab_size * N_EMBD);

    for (int li = 0; li < N_LAYER; li++) {
        init_matrix(W_attn_wq[li], N_EMBD * N_EMBD, 0.08);
        init_matrix(W_attn_wk[li], N_EMBD * N_EMBD, 0.08);
        init_matrix(W_attn_wv[li], N_EMBD * N_EMBD, 0.08);
        init_matrix(W_attn_wo[li], N_EMBD * N_EMBD, 0.08);
        init_matrix(W_mlp_fc1[li], MLP_DIM * N_EMBD, 0.08);
        init_matrix(W_mlp_fc2[li], N_EMBD * MLP_DIM, 0.08);
        register_matrix(W_attn_wq[li], G_attn_wq[li], N_EMBD * N_EMBD);
        register_matrix(W_attn_wk[li], G_attn_wk[li], N_EMBD * N_EMBD);
        register_matrix(W_attn_wv[li], G_attn_wv[li], N_EMBD * N_EMBD);
        register_matrix(W_attn_wo[li], G_attn_wo[li], N_EMBD * N_EMBD);
        register_matrix(W_mlp_fc1[li], G_mlp_fc1[li], MLP_DIM * N_EMBD);
        register_matrix(W_mlp_fc2[li], G_mlp_fc2[li], N_EMBD * MLP_DIM);
    }

    memset(adam_m, 0, sizeof(adam_m));
    memset(adam_v, 0, sizeof(adam_v));
    printf("num params: %d\n", num_params);
}

static void zero_grads(void) {
    memset(G_wte,     0, sizeof(G_wte));
    memset(G_wpe,     0, sizeof(G_wpe));
    memset(G_lm_head, 0, sizeof(G_lm_head));
    for (int li = 0; li < N_LAYER; li++) {
        memset(G_attn_wq[li], 0, sizeof(G_attn_wq[li]));
        memset(G_attn_wk[li], 0, sizeof(G_attn_wk[li]));
        memset(G_attn_wv[li], 0, sizeof(G_attn_wv[li]));
        memset(G_attn_wo[li], 0, sizeof(G_attn_wo[li]));
        memset(G_mlp_fc1[li], 0, sizeof(G_mlp_fc1[li]));
        memset(G_mlp_fc2[li], 0, sizeof(G_mlp_fc2[li]));
    }
}

/* ── Stored Activations (for backward pass) ────────────────────── */

typedef struct {
    /* Embedding */
    double x_embed[N_EMBD];        /* tok_emb + pos_emb */
    double x_rms0[N_EMBD];         /* after initial rmsnorm */
    double rms0_scale;              /* 1/rms for initial rmsnorm */

    /* Per-layer (we store for layer 0; extend arrays for N_LAYER>1) */
    double x_res_attn[N_EMBD];     /* residual save point before attention */
    double xn_attn[N_EMBD];        /* after pre-attn rmsnorm */
    double rms_attn_scale;
    double q[N_EMBD], k[N_EMBD], v[N_EMBD];
    double attn_weights[N_HEAD][BLOCK_SIZE];
    double attn_concat[N_EMBD];    /* concatenated head outputs */
    double x_post_attn[N_EMBD];    /* after Wo + residual */

    double x_res_mlp[N_EMBD];      /* residual save point before MLP */
    double xn_mlp[N_EMBD];         /* after pre-mlp rmsnorm */
    double rms_mlp_scale;
    double h_pre_relu[MLP_DIM];
    double h_post_relu[MLP_DIM];
    double x_post_mlp[N_EMBD];     /* after fc2 + residual (final x) */

    /* Output */
    double logits[MAX_VOCAB];
    double probs[MAX_VOCAB];
} Activations;

static Activations act[BLOCK_SIZE];

/* KV cache: stored separately for convenient cross-position access */
static double kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static double kv_vals[N_LAYER][BLOCK_SIZE][N_EMBD];

/* Accumulated gradients for KV entries (from attention backward) */
static double grad_k_accum[BLOCK_SIZE][N_EMBD];
static double grad_v_accum[BLOCK_SIZE][N_EMBD];

/* ── BLAS-accelerated primitives ───────────────────────────────── */

/*
 * Forward: y = W @ x
 *   W is (nout × nin), row-major. x is nin-vector. y is nout-vector.
 */
static inline void blas_matvec(const double *W, const double *x,
                               double *y, int nout, int nin) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                nout, nin, 1.0, W, nin, x, 1, 0.0, y, 1);
}

/*
 * Backward input: grad_x = W^T @ grad_y
 *   W is (nout × nin). grad_y is nout-vector. grad_x is nin-vector.
 */
static inline void blas_matvec_T(const double *W, const double *grad_y,
                                 double *grad_x, int nout, int nin) {
    cblas_dgemv(CblasRowMajor, CblasTrans,
                nout, nin, 1.0, W, nin, grad_y, 1, 0.0, grad_x, 1);
}

/*
 * Backward weight: grad_W += grad_y ⊗ x^T  (rank-1 update)
 *   grad_W is (nout × nin). grad_y is nout-vector. x is nin-vector.
 */
static inline void blas_outer_accum(double *grad_W, const double *grad_y,
                                    const double *x, int nout, int nin) {
    cblas_dger(CblasRowMajor, nout, nin, 1.0,
              grad_y, 1, x, 1, grad_W, nin);
}

/* ── Forward RMSNorm ───────────────────────────────────────────── */

static double fwd_rmsnorm(const double *x, double *y, int n) {
    double ms = cblas_ddot(n, x, 1, x, 1) / n + 1e-5;
    double scale = 1.0 / sqrt(ms);
    for (int i = 0; i < n; i++) y[i] = x[i] * scale;
    return scale;
}

/* ── Backward RMSNorm ──────────────────────────────────────────── */
/*
 * Given y = x * scale, where scale = (mean(x²) + eps)^(-0.5):
 *   grad_x[i] = scale * (grad_y[i] - (scale² / n) * x[i] * dot(x, grad_y))
 */
static void bwd_rmsnorm(const double *x, const double *grad_y,
                         double *grad_x, double scale, int n) {
    double dot_xg = cblas_ddot(n, x, 1, grad_y, 1);
    double coeff = scale * scale / n * dot_xg;
    for (int i = 0; i < n; i++)
        grad_x[i] = scale * (grad_y[i] - coeff * x[i]);
}

/* ── Forward Softmax ───────────────────────────────────────────── */

static void fwd_softmax(const double *logits, double *probs, int n) {
    double max_v = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_v) max_v = logits[i];
    double sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = exp(logits[i] - max_v);
        sum += probs[i];
    }
    double inv = 1.0 / sum;
    cblas_dscal(n, inv, probs, 1);
}

/* ── Backward Softmax ──────────────────────────────────────────── */
/*
 * grad_logits[i] = probs[i] * (grad_probs[i] - dot(grad_probs, probs))
 */
static void bwd_softmax(const double *probs, const double *grad_probs,
                         double *grad_logits, int n) {
    double dot = cblas_ddot(n, grad_probs, 1, probs, 1);
    for (int i = 0; i < n; i++)
        grad_logits[i] = probs[i] * (grad_probs[i] - dot);
}

/* ── Full Forward Pass (one token) ─────────────────────────────── */

static void forward_token(int token_id, int pos_id, int pos_idx) {
    Activations *a = &act[pos_idx];

    /* Token embedding + position embedding */
    const double *te = &W_wte[token_id * N_EMBD];
    const double *pe = &W_wpe[pos_id   * N_EMBD];
    for (int j = 0; j < N_EMBD; j++)
        a->x_embed[j] = te[j] + pe[j];

    /* Initial RMSNorm */
    a->rms0_scale = fwd_rmsnorm(a->x_embed, a->x_rms0, N_EMBD);

    double *x = a->x_rms0;  /* current hidden state */

    for (int li = 0; li < N_LAYER; li++) {
        /* ── Attention block ── */
        memcpy(a->x_res_attn, x, N_EMBD * sizeof(double));

        /* Pre-attention RMSNorm */
        a->rms_attn_scale = fwd_rmsnorm(x, a->xn_attn, N_EMBD);

        /* Q, K, V via BLAS */
        blas_matvec(W_attn_wq[li], a->xn_attn, a->q, N_EMBD, N_EMBD);
        blas_matvec(W_attn_wk[li], a->xn_attn, a->k, N_EMBD, N_EMBD);
        blas_matvec(W_attn_wv[li], a->xn_attn, a->v, N_EMBD, N_EMBD);

        /* Store in KV cache */
        memcpy(kv_keys[li][pos_idx], a->k, N_EMBD * sizeof(double));
        memcpy(kv_vals[li][pos_idx], a->v, N_EMBD * sizeof(double));

        int n_kv = pos_idx + 1;

        /* Multi-head attention */
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;

            /* Attention scores: score[t] = q_h · k_h[t] / sqrt(head_dim) */
            double scores[BLOCK_SIZE];
            double inv_sqrt_hd = 1.0 / sqrt((double)HEAD_DIM);
            for (int t = 0; t < n_kv; t++) {
                scores[t] = cblas_ddot(HEAD_DIM, &a->q[hs], 1,
                                       &kv_keys[li][t][hs], 1) * inv_sqrt_hd;
            }

            /* Softmax */
            fwd_softmax(scores, a->attn_weights[h], n_kv);

            /* Weighted sum of values */
            memset(&a->attn_concat[hs], 0, HEAD_DIM * sizeof(double));
            for (int t = 0; t < n_kv; t++) {
                cblas_daxpy(HEAD_DIM, a->attn_weights[h][t],
                           &kv_vals[li][t][hs], 1, &a->attn_concat[hs], 1);
            }
        }

        /* Output projection: Wo @ attn_concat */
        double wo_out[N_EMBD];
        blas_matvec(W_attn_wo[li], a->attn_concat, wo_out, N_EMBD, N_EMBD);

        /* Residual */
        for (int j = 0; j < N_EMBD; j++)
            a->x_post_attn[j] = wo_out[j] + a->x_res_attn[j];

        x = a->x_post_attn;

        /* ── MLP block ── */
        memcpy(a->x_res_mlp, x, N_EMBD * sizeof(double));

        /* Pre-MLP RMSNorm */
        a->rms_mlp_scale = fwd_rmsnorm(x, a->xn_mlp, N_EMBD);

        /* fc1: MLP_DIM × N_EMBD */
        blas_matvec(W_mlp_fc1[li], a->xn_mlp, a->h_pre_relu, MLP_DIM, N_EMBD);

        /* ReLU */
        for (int j = 0; j < MLP_DIM; j++)
            a->h_post_relu[j] = a->h_pre_relu[j] > 0 ? a->h_pre_relu[j] : 0.0;

        /* fc2: N_EMBD × MLP_DIM */
        double fc2_out[N_EMBD];
        blas_matvec(W_mlp_fc2[li], a->h_post_relu, fc2_out, N_EMBD, MLP_DIM);

        /* Residual */
        for (int j = 0; j < N_EMBD; j++)
            a->x_post_mlp[j] = fc2_out[j] + a->x_res_mlp[j];

        x = a->x_post_mlp;
    }

    /* Final logits: lm_head @ x */
    blas_matvec(W_lm_head, x, a->logits, vocab_size, N_EMBD);

    /* Softmax for loss */
    fwd_softmax(a->logits, a->probs, vocab_size);
}

/* ── Full Backward Pass ────────────────────────────────────────── */

static void backward_sequence(const int *tokens, int n) {
    /*
     * Process positions in reverse. Attention backward at position t
     * produces gradients for K and V at positions 0..t, which we
     * accumulate in grad_k_accum / grad_v_accum.
     */

    /* Zero KV gradient accumulators */
    memset(grad_k_accum, 0, sizeof(grad_k_accum));
    memset(grad_v_accum, 0, sizeof(grad_v_accum));

    double inv_n = 1.0 / (double)n;

    for (int pos = n - 1; pos >= 0; pos--) {
        Activations *a = &act[pos];
        int target = tokens[pos + 1];

        /* ── Loss gradient: d(-log(prob[target]))/d(logits) ── */
        /* For cross-entropy + softmax: grad_logits = probs - one_hot(target) */
        double grad_logits[MAX_VOCAB];
        for (int i = 0; i < vocab_size; i++)
            grad_logits[i] = (a->probs[i] - (i == target ? 1.0 : 0.0)) * inv_n;

        /* ── Backward through lm_head ── */
        /* grad_x_final = W_lm_head^T @ grad_logits */
        double *x_final = a->x_post_mlp;  /* output of last layer */
        double grad_x[N_EMBD];
        blas_matvec_T(W_lm_head, grad_logits, grad_x, vocab_size, N_EMBD);

        /* grad_W_lm_head += grad_logits ⊗ x_final^T */
        blas_outer_accum(G_lm_head, grad_logits, x_final, vocab_size, N_EMBD);

        /* ── Backward through layers (reverse) ── */
        for (int li = N_LAYER - 1; li >= 0; li--) {
            int n_kv = pos + 1;

            /* ── MLP backward ── */
            /* grad_x = grad through residual: one copy passes through, one through MLP */
            double grad_x_res_mlp[N_EMBD];
            memcpy(grad_x_res_mlp, grad_x, N_EMBD * sizeof(double));

            /* Backward fc2: grad_h_post_relu = W_fc2^T @ grad_x */
            double grad_h_post_relu[MLP_DIM];
            blas_matvec_T(W_mlp_fc2[li], grad_x, grad_h_post_relu, N_EMBD, MLP_DIM);

            /* grad_W_fc2 += grad_x ⊗ h_post_relu^T */
            blas_outer_accum(G_mlp_fc2[li], grad_x, a->h_post_relu, N_EMBD, MLP_DIM);

            /* Backward ReLU */
            double grad_h_pre_relu[MLP_DIM];
            for (int j = 0; j < MLP_DIM; j++)
                grad_h_pre_relu[j] = a->h_pre_relu[j] > 0 ? grad_h_post_relu[j] : 0.0;

            /* Backward fc1: grad_xn_mlp = W_fc1^T @ grad_h_pre_relu */
            double grad_xn_mlp[N_EMBD];
            blas_matvec_T(W_mlp_fc1[li], grad_h_pre_relu, grad_xn_mlp, MLP_DIM, N_EMBD);

            /* grad_W_fc1 += grad_h_pre_relu ⊗ xn_mlp^T */
            blas_outer_accum(G_mlp_fc1[li], grad_h_pre_relu, a->xn_mlp, MLP_DIM, N_EMBD);

            /* Backward pre-MLP RMSNorm */
            double grad_x_pre_mlp[N_EMBD];
            bwd_rmsnorm(a->x_res_mlp, grad_xn_mlp, grad_x_pre_mlp,
                       a->rms_mlp_scale, N_EMBD);

            /* Add residual gradient */
            for (int j = 0; j < N_EMBD; j++)
                grad_x[j] = grad_x_pre_mlp[j] + grad_x_res_mlp[j];

            /* ── Attention backward ── */
            double grad_x_res_attn[N_EMBD];
            memcpy(grad_x_res_attn, grad_x, N_EMBD * sizeof(double));

            /* Backward Wo: grad_attn_concat = W_wo^T @ grad_x */
            double grad_attn_concat[N_EMBD];
            blas_matvec_T(W_attn_wo[li], grad_x, grad_attn_concat, N_EMBD, N_EMBD);

            /* grad_W_wo += grad_x ⊗ attn_concat^T */
            blas_outer_accum(G_attn_wo[li], grad_x, a->attn_concat, N_EMBD, N_EMBD);

            /* Backward multi-head attention */
            double grad_q[N_EMBD];
            memset(grad_q, 0, sizeof(grad_q));

            double inv_sqrt_hd = 1.0 / sqrt((double)HEAD_DIM);

            for (int h = 0; h < N_HEAD; h++) {
                int hs = h * HEAD_DIM;
                const double *weights = a->attn_weights[h];

                /* Backward weighted sum: out[j] = Σ_t w[t] * v[t][j] */
                double grad_weights[BLOCK_SIZE];
                memset(grad_weights, 0, n_kv * sizeof(double));

                for (int t = 0; t < n_kv; t++) {
                    /* grad_weights[t] = dot(grad_out_h, v_h[t]) */
                    grad_weights[t] = cblas_ddot(HEAD_DIM,
                        &grad_attn_concat[hs], 1, &kv_vals[li][t][hs], 1);

                    /* grad_v[t][hs..] += weights[t] * grad_out_h */
                    cblas_daxpy(HEAD_DIM, weights[t],
                               &grad_attn_concat[hs], 1,
                               &grad_v_accum[t][hs], 1);
                }

                /* Backward softmax: grad_scores from grad_weights */
                double grad_scores[BLOCK_SIZE];
                bwd_softmax(weights, grad_weights, grad_scores, n_kv);

                /* Scale by 1/sqrt(head_dim) */
                cblas_dscal(n_kv, inv_sqrt_hd, grad_scores, 1);

                /* Backward through attention dot products */
                for (int t = 0; t < n_kv; t++) {
                    /* grad_q_h += grad_scores[t] * k_h[t] */
                    cblas_daxpy(HEAD_DIM, grad_scores[t],
                               &kv_keys[li][t][hs], 1, &grad_q[hs], 1);

                    /* grad_k[t][hs..] += grad_scores[t] * q_h */
                    cblas_daxpy(HEAD_DIM, grad_scores[t],
                               &a->q[hs], 1, &grad_k_accum[t][hs], 1);
                }
            }

            /* Backward Q, K, V projections */
            /* The gradient for K and V at this position comes from:
             * 1. grad_k_accum[pos] / grad_v_accum[pos] (from attention at pos and later)
             * 2. We also need to backward through the projection to get grad_xn_attn
             */
            double grad_xn_attn[N_EMBD];
            memset(grad_xn_attn, 0, sizeof(grad_xn_attn));

            /* Backward Wq: grad_xn += Wq^T @ grad_q */
            double tmp[N_EMBD];
            blas_matvec_T(W_attn_wq[li], grad_q, tmp, N_EMBD, N_EMBD);
            cblas_daxpy(N_EMBD, 1.0, tmp, 1, grad_xn_attn, 1);
            /* grad_Wq += grad_q ⊗ xn_attn^T */
            blas_outer_accum(G_attn_wq[li], grad_q, a->xn_attn, N_EMBD, N_EMBD);

            /* Backward Wk: grad_xn += Wk^T @ grad_k_accum[pos] */
            blas_matvec_T(W_attn_wk[li], grad_k_accum[pos], tmp, N_EMBD, N_EMBD);
            cblas_daxpy(N_EMBD, 1.0, tmp, 1, grad_xn_attn, 1);
            /* grad_Wk += grad_k ⊗ xn_attn^T */
            blas_outer_accum(G_attn_wk[li], grad_k_accum[pos], a->xn_attn, N_EMBD, N_EMBD);

            /* Backward Wv: grad_xn += Wv^T @ grad_v_accum[pos] */
            blas_matvec_T(W_attn_wv[li], grad_v_accum[pos], tmp, N_EMBD, N_EMBD);
            cblas_daxpy(N_EMBD, 1.0, tmp, 1, grad_xn_attn, 1);
            /* grad_Wv += grad_v ⊗ xn_attn^T */
            blas_outer_accum(G_attn_wv[li], grad_v_accum[pos], a->xn_attn, N_EMBD, N_EMBD);

            /* Backward pre-attn RMSNorm */
            double grad_x_pre_attn[N_EMBD];
            bwd_rmsnorm(a->x_res_attn, grad_xn_attn, grad_x_pre_attn,
                       a->rms_attn_scale, N_EMBD);

            /* Add residual gradient */
            for (int j = 0; j < N_EMBD; j++)
                grad_x[j] = grad_x_pre_attn[j] + grad_x_res_attn[j];
        }

        /* ── Backward through initial RMSNorm ── */
        double grad_embed[N_EMBD];
        bwd_rmsnorm(a->x_embed, grad_x, grad_embed, a->rms0_scale, N_EMBD);

        /* ── Backward through embedding lookup ── */
        int token_id = tokens[pos];
        /* grad_wte[token_id] += grad_embed */
        cblas_daxpy(N_EMBD, 1.0, grad_embed, 1,
                   &G_wte[token_id * N_EMBD], 1);
        /* grad_wpe[pos_id] += grad_embed */
        cblas_daxpy(N_EMBD, 1.0, grad_embed, 1,
                   &G_wpe[pos * N_EMBD], 1);
    }
}

/* ── Adam Optimizer Step ───────────────────────────────────────── */

static void adam_step(int step) {
    double lr_t = LR * (1.0 - (double)step / NUM_STEPS);
    double bc1 = 1.0 - pow(BETA1, step + 1);
    double bc2 = 1.0 - pow(BETA2, step + 1);

    for (int i = 0; i < num_params; i++) {
        double g = *all_G[i];
        adam_m[i] = BETA1 * adam_m[i] + (1.0 - BETA1) * g;
        adam_v[i] = BETA2 * adam_v[i] + (1.0 - BETA2) * g * g;
        double m_hat = adam_m[i] / bc1;
        double v_hat = adam_v[i] / bc2;
        *all_W[i] -= lr_t * m_hat / (sqrt(v_hat) + EPS_ADAM);
    }
}

/* ── Training Loop ─────────────────────────────────────────────── */

static void train(void) {
    for (int step = 0; step < NUM_STEPS; step++) {
        int di = doc_order[step % num_docs];
        const char *doc = docs[di];
        int tokens[MAX_SEQ];
        int tlen = 0;
        tokens[tlen++] = BOS;
        for (int i = 0; doc[i] && tlen < MAX_SEQ - 1; i++)
            tokens[tlen++] = char_to_id(doc[i]);
        tokens[tlen++] = BOS;

        int n = tlen - 1;
        if (n > BLOCK_SIZE) n = BLOCK_SIZE;
        if (n < 1) continue;

        /* Forward all positions */
        for (int pos = 0; pos < n; pos++)
            forward_token(tokens[pos], pos, pos);

        /* Compute loss */
        double loss = 0;
        for (int pos = 0; pos < n; pos++) {
            double p = act[pos].probs[tokens[pos + 1]];
            if (p < 1e-30) p = 1e-30;
            loss += -log(p);
        }
        loss /= n;

        /* Backward */
        zero_grads();
        backward_sequence(tokens, n);

        /* Adam */
        adam_step(step);

        printf("\rstep %4d / %4d | loss %.4f", step + 1, NUM_STEPS, loss);
        fflush(stdout);
    }
    printf("\n");
}

/* ── Inference ─────────────────────────────────────────────────── */

static int weighted_choice(const double *weights, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += weights[i];
    double r = rng_uniform() * sum;
    double acc = 0;
    for (int i = 0; i < n; i++) {
        acc += weights[i];
        if (r <= acc) return i;
    }
    return n - 1;
}

static void inference(int num_samples) {
    printf("--- inference (new, hallucinated names) ---\n");
    for (int s = 0; s < num_samples; s++) {
        int token_id = BOS;
        char name[BLOCK_SIZE + 1];
        int name_len = 0;

        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            forward_token(token_id, pos, pos);

            /* Apply temperature to logits and softmax */
            double probs[MAX_VOCAB];
            double max_l = act[pos].logits[0];
            for (int i = 1; i < vocab_size; i++)
                if (act[pos].logits[i] > max_l) max_l = act[pos].logits[i];
            double sum = 0;
            for (int i = 0; i < vocab_size; i++) {
                probs[i] = exp(act[pos].logits[i] / TEMPERATURE - max_l / TEMPERATURE);
                sum += probs[i];
            }
            cblas_dscal(vocab_size, 1.0 / sum, probs, 1);

            token_id = weighted_choice(probs, vocab_size);
            if (token_id == BOS) break;
            if (name_len < BLOCK_SIZE)
                name[name_len++] = uchars[token_id];
        }
        name[name_len] = 0;
        printf("sample %2d: %s\n", s + 1, name);
    }
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *data_path = "input.txt";
    FILE *test = fopen(data_path, "r");
    if (!test) {
        printf("Downloading names.txt...\n");
        system("curl -sfL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt");
    } else {
        fclose(test);
    }

    rng_seed(42);
    load_dataset(data_path);
    init_params();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    train();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("Training time: %.4f seconds\n", elapsed);

    inference(20);
    return 0;
}
