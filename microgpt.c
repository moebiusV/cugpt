/*
 * microgpt.c — Optimal C translation of Karpathy's microgpt.py
 *
 * Tape-based reverse-mode autograd with arena bump allocator.
 * Single file, zero dependencies beyond libc/libm.
 * Faithful to the original algorithm; everything else is efficiency.
 *
 * Build:  cc -O2 -march=native -o microgpt microgpt.c -lm
 * Run:    ./microgpt              (downloads names.txt if missing)
 *
 * Architecture: 1-layer GPT-2 variant (RMSNorm, ReLU, no biases)
 *   vocab_size=27, n_embd=16, n_head=4, block_size=16
 *   4,192 parameters, character-level tokenizer on names
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Configuration ─────────────────────────────────────────────── */

#define N_LAYER    1
#define N_EMBD     16
#define BLOCK_SIZE 16
#define N_HEAD     4
#define HEAD_DIM   (N_EMBD / N_HEAD)   /* 4 */
#define MLP_DIM    (4 * N_EMBD)        /* 64 */
#define NUM_STEPS  1000
#define LR         0.01
#define BETA1      0.85
#define BETA2      0.99
#define EPS_ADAM   1e-8
#define TEMPERATURE 0.5

/* Maximum tokens in a single name + 2 BOS sentinels */
#define MAX_SEQ    (BLOCK_SIZE + 2)

/* ── Tape-based Autograd ───────────────────────────────────────── */
/*
 * Instead of building a DAG of heap-allocated Value nodes (as the
 * Python original does), we use a Wengert tape: a flat array where
 * each entry records the operation, operand indices, and a cached
 * local gradient. Backward is just a reverse linear scan.
 *
 * This eliminates: recursive topo-sort, pointer chasing, hash-set
 * visited checks, and per-node heap allocation. The gist benchmarks
 * show this alone gives ~4x speedup even in Python; in C with an
 * arena it's the dominant optimization.
 */

/* Operation tags */
enum {
    OP_LEAF = 0,   /* parameter or constant — no parents */
    OP_ADD,        /* c = a + b */
    OP_MUL,        /* c = a * b */
    OP_POW,        /* c = a ^ (constant exponent) */
    OP_LOG,        /* c = log(a) */
    OP_EXP,        /* c = exp(a) */
    OP_RELU,       /* c = max(0, a) */
};

typedef struct {
    double val;          /* forward value */
    double grad;         /* accumulated gradient (set during backward) */
    int    op;           /* operation tag */
    int    parent_a;     /* index of first operand (-1 if none) */
    int    parent_b;     /* index of second operand (-1 if none) */
    double local_a;      /* ∂this/∂parent_a */
    double local_b;      /* ∂this/∂parent_b */
} TapeEntry;

/* Arena: pre-allocated tape buffer, reset every training step */
#define TAPE_CAP (1 << 22)   /* ~4M entries, ~192 MB — plenty for 16×16 */
static TapeEntry tape[TAPE_CAP];
static int       tape_len;

static inline void tape_reset(void) { tape_len = 0; }

static inline int tape_leaf(double v) {
    int i = tape_len++;
    tape[i] = (TapeEntry){v, 0.0, OP_LEAF, -1, -1, 0.0, 0.0};
    return i;
}

static inline int tape_add(int a, int b) {
    int i = tape_len++;
    tape[i] = (TapeEntry){
        tape[a].val + tape[b].val, 0.0,
        OP_ADD, a, b, 1.0, 1.0
    };
    return i;
}

static inline int tape_mul(int a, int b) {
    int i = tape_len++;
    tape[i] = (TapeEntry){
        tape[a].val * tape[b].val, 0.0,
        OP_MUL, a, b, tape[b].val, tape[a].val
    };
    return i;
}

static inline int tape_pow_const(int a, double e) {
    double v = pow(tape[a].val, e);
    int i = tape_len++;
    tape[i] = (TapeEntry){
        v, 0.0,
        OP_POW, a, -1, e * pow(tape[a].val, e - 1.0), 0.0
    };
    return i;
}

static inline int tape_log(int a) {
    int i = tape_len++;
    tape[i] = (TapeEntry){
        log(tape[a].val), 0.0,
        OP_LOG, a, -1, 1.0 / tape[a].val, 0.0
    };
    return i;
}

static inline int tape_exp(int a) {
    double ev = exp(tape[a].val);
    int i = tape_len++;
    tape[i] = (TapeEntry){
        ev, 0.0,
        OP_EXP, a, -1, ev, 0.0
    };
    return i;
}

static inline int tape_relu(int a) {
    double v = tape[a].val;
    int i = tape_len++;
    tape[i] = (TapeEntry){
        v > 0 ? v : 0.0, 0.0,
        OP_RELU, a, -1, v > 0 ? 1.0 : 0.0, 0.0
    };
    return i;
}

/* Derived: a - b  =  a + (b * -1) */
static inline int tape_sub(int a, int b) {
    int neg1 = tape_leaf(-1.0);
    int nb = tape_mul(b, neg1);
    return tape_add(a, nb);
}

/* Derived: a / b  =  a * b^(-1) */
static inline int tape_div(int a, int b) {
    int inv = tape_pow_const(b, -1.0);
    return tape_mul(a, inv);
}

/* Backward: single reverse pass over the tape */
static void tape_backward(int loss_idx) {
    tape[loss_idx].grad = 1.0;
    for (int i = loss_idx; i >= 0; i--) {
        double g = tape[i].grad;
        if (g == 0.0) continue;
        if (tape[i].parent_a >= 0)
            tape[tape[i].parent_a].grad += tape[i].local_a * g;
        if (tape[i].parent_b >= 0)
            tape[tape[i].parent_b].grad += tape[i].local_b * g;
    }
}

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
    /* SplitMix64 to initialize state */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}

/* Uniform [0, 1) */
static inline double rng_uniform(void) {
    return (rng_next() >> 11) * 0x1.0p-53;
}

/* Gaussian via Box-Muller */
static double rng_gauss(double mean, double std) {
    double u1 = rng_uniform(), u2 = rng_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Shuffle array of indices */
static void shuffle(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_uniform() * (i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ── Dataset ───────────────────────────────────────────────────── */

#define MAX_DOCS   40000
#define MAX_DOC_LEN 20

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int  doc_order[MAX_DOCS];  /* shuffled indices */
static int  num_docs;

/* Vocabulary: sorted unique chars → token ids 0..n-1, BOS = n */
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
        fprintf(stderr, "Cannot open %s — download names.txt first.\n", path);
        fprintf(stderr, "  curl -fL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt\n");
        exit(1);
    }
    char line[256];
    /* Collect unique chars */
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

    /* Build sorted unique char list */
    int nc = 0;
    for (int c = 0; c < 256; c++)
        if (char_seen[c]) uchars[nc++] = (char)c;
    BOS = nc;
    vocab_size = nc + 1;

    /* Shuffle docs */
    for (int i = 0; i < num_docs; i++) doc_order[i] = i;
    shuffle(doc_order, num_docs);

    printf("num docs: %d\n", num_docs);
    printf("vocab size: %d\n", vocab_size);
}

/* ── Parameters ────────────────────────────────────────────────── */
/*
 * All parameter values stored in flat arrays.  Each param has a
 * persistent double (its value) and Adam buffers.  During each
 * forward pass we create a tape leaf pointing to the param's value;
 * after backward we read the gradient from that leaf.
 */

/* Total params: wte + wpe + lm_head + per-layer(Wq,Wk,Wv,Wo,fc1,fc2)
 *   wte:     vocab_size * N_EMBD  =  27 * 16 = 432
 *   wpe:     BLOCK_SIZE * N_EMBD  =  16 * 16 = 256
 *   lm_head: vocab_size * N_EMBD  =  27 * 16 = 432
 *   Per layer:
 *     attn_wq: N_EMBD * N_EMBD = 256
 *     attn_wk: N_EMBD * N_EMBD = 256
 *     attn_wv: N_EMBD * N_EMBD = 256
 *     attn_wo: N_EMBD * N_EMBD = 256
 *     mlp_fc1: MLP_DIM * N_EMBD = 1024
 *     mlp_fc2: N_EMBD * MLP_DIM = 1024
 *   Layer total: 3072
 *   Grand total: 432 + 256 + 432 + 3072 = 4192
 */

#define MAX_PARAMS 8192

static double param_val[MAX_PARAMS];
static double adam_m[MAX_PARAMS];
static double adam_v[MAX_PARAMS];
static int    param_tape_idx[MAX_PARAMS];  /* tape index for this step */
static int    num_params;

/* Offsets into param_val[] for each weight matrix */
static int off_wte, off_wpe, off_lm_head;
static int off_attn_wq[N_LAYER], off_attn_wk[N_LAYER];
static int off_attn_wv[N_LAYER], off_attn_wo[N_LAYER];
static int off_mlp_fc1[N_LAYER], off_mlp_fc2[N_LAYER];

static int alloc_matrix(int rows, int cols, double std) {
    int off = num_params;
    for (int i = 0; i < rows * cols; i++)
        param_val[num_params++] = rng_gauss(0.0, std);
    return off;
}

static void init_params(void) {
    num_params = 0;
    off_wte     = alloc_matrix(vocab_size, N_EMBD, 0.08);
    off_wpe     = alloc_matrix(BLOCK_SIZE, N_EMBD, 0.08);
    off_lm_head = alloc_matrix(vocab_size, N_EMBD, 0.08);
    for (int li = 0; li < N_LAYER; li++) {
        off_attn_wq[li] = alloc_matrix(N_EMBD, N_EMBD, 0.08);
        off_attn_wk[li] = alloc_matrix(N_EMBD, N_EMBD, 0.08);
        off_attn_wv[li] = alloc_matrix(N_EMBD, N_EMBD, 0.08);
        off_attn_wo[li] = alloc_matrix(N_EMBD, N_EMBD, 0.08);
        off_mlp_fc1[li] = alloc_matrix(MLP_DIM, N_EMBD, 0.08);
        off_mlp_fc2[li] = alloc_matrix(N_EMBD, MLP_DIM, 0.08);
    }
    memset(adam_m, 0, sizeof(adam_m));
    memset(adam_v, 0, sizeof(adam_v));
    printf("num params: %d\n", num_params);
}

/* Put all params on the tape as leaves; record their tape indices */
static void params_to_tape(void) {
    for (int i = 0; i < num_params; i++)
        param_tape_idx[i] = tape_leaf(param_val[i]);
}

/* ── Model: forward pass building the tape ─────────────────────── */

/* linear(x, W): y[row] = Σ_col W[row][col] * x[col]
 *   W is stored row-major at offset `w_off`, shape (nout, nin)
 *   x is an array of `nin` tape indices
 *   y is written to `out`, an array of `nout` tape indices
 */
static void linear(const int *x, int nin, int w_off, int nout, int *out) {
    for (int r = 0; r < nout; r++) {
        int acc = tape_mul(param_tape_idx[w_off + r * nin], x[0]);
        for (int c = 1; c < nin; c++)
            acc = tape_add(acc, tape_mul(param_tape_idx[w_off + r * nin + c], x[c]));
        out[r] = acc;
    }
}

/* softmax: returns tape indices in `out` */
static void softmax(const int *logits, int n, int *out) {
    /* Find max for numerical stability (using forward values) */
    double max_val = tape[logits[0]].val;
    for (int i = 1; i < n; i++)
        if (tape[logits[i]].val > max_val) max_val = tape[logits[i]].val;

    int max_node = tape_leaf(max_val);
    int exps[128];
    for (int i = 0; i < n; i++)
        exps[i] = tape_exp(tape_sub(logits[i], max_node));

    /* total = sum of exps */
    int total = exps[0];
    for (int i = 1; i < n; i++)
        total = tape_add(total, exps[i]);

    for (int i = 0; i < n; i++)
        out[i] = tape_div(exps[i], total);
}

/* rmsnorm(x) */
static void rmsnorm(const int *x, int n, int *out) {
    /* ms = mean of squares */
    int ms = tape_mul(x[0], x[0]);
    for (int i = 1; i < n; i++)
        ms = tape_add(ms, tape_mul(x[i], x[i]));
    int len_node = tape_leaf((double)n);
    ms = tape_div(ms, len_node);

    /* scale = (ms + 1e-5) ^ -0.5 */
    int eps = tape_leaf(1e-5);
    int scale = tape_pow_const(tape_add(ms, eps), -0.5);

    for (int i = 0; i < n; i++)
        out[i] = tape_mul(x[i], scale);
}

/* KV cache: stored as tape indices */
static int kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static int kv_vals[N_LAYER][BLOCK_SIZE][N_EMBD];
static int kv_len[N_LAYER];

static void kv_reset(void) {
    for (int li = 0; li < N_LAYER; li++) kv_len[li] = 0;
}

/* GPT forward for a single token at a given position.
 * Returns tape indices for logits in `logits_out` (size vocab_size).
 */
static void gpt_forward(int token_id, int pos_id, int *logits_out) {
    int x[N_EMBD], tmp[N_EMBD], tmp2[MLP_DIM];

    /* Token + position embedding */
    for (int j = 0; j < N_EMBD; j++)
        x[j] = tape_add(
            param_tape_idx[off_wte + token_id * N_EMBD + j],
            param_tape_idx[off_wpe + pos_id   * N_EMBD + j]
        );

    /* Initial RMSNorm */
    rmsnorm(x, N_EMBD, x);

    for (int li = 0; li < N_LAYER; li++) {
        /* Save residual */
        int x_res[N_EMBD];
        memcpy(x_res, x, sizeof(x_res));

        /* Pre-attention RMSNorm */
        int xn[N_EMBD];
        rmsnorm(x, N_EMBD, xn);

        /* Q, K, V projections */
        int q[N_EMBD], k[N_EMBD], v[N_EMBD];
        linear(xn, N_EMBD, off_attn_wq[li], N_EMBD, q);
        linear(xn, N_EMBD, off_attn_wk[li], N_EMBD, k);
        linear(xn, N_EMBD, off_attn_wv[li], N_EMBD, v);

        /* Store K, V in cache */
        int t = kv_len[li];
        memcpy(kv_keys[li][t], k, sizeof(k));
        memcpy(kv_vals[li][t], v, sizeof(v));
        kv_len[li]++;

        /* Multi-head attention */
        int x_attn[N_EMBD];
        int attn_idx = 0;
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;

            /* Compute attention logits: score[t] = q_h · k_h[t] / sqrt(head_dim) */
            int n_kv = kv_len[li];
            int attn_logits[BLOCK_SIZE];
            int scale_node = tape_leaf(1.0 / sqrt((double)HEAD_DIM));

            for (int ti = 0; ti < n_kv; ti++) {
                int dot = tape_mul(q[hs], kv_keys[li][ti][hs]);
                for (int j = 1; j < HEAD_DIM; j++)
                    dot = tape_add(dot, tape_mul(q[hs + j], kv_keys[li][ti][hs + j]));
                attn_logits[ti] = tape_mul(dot, scale_node);
            }

            /* Softmax over attention logits */
            int attn_weights[BLOCK_SIZE];
            softmax(attn_logits, n_kv, attn_weights);

            /* Weighted sum of values */
            for (int j = 0; j < HEAD_DIM; j++) {
                int acc = tape_mul(attn_weights[0], kv_vals[li][0][hs + j]);
                for (int ti = 1; ti < n_kv; ti++)
                    acc = tape_add(acc, tape_mul(attn_weights[ti], kv_vals[li][ti][hs + j]));
                x_attn[attn_idx++] = acc;
            }
        }

        /* Output projection + residual */
        int attn_out[N_EMBD];
        linear(x_attn, N_EMBD, off_attn_wo[li], N_EMBD, attn_out);
        for (int j = 0; j < N_EMBD; j++)
            x[j] = tape_add(attn_out[j], x_res[j]);

        /* MLP block */
        memcpy(x_res, x, sizeof(x_res));
        int xn2[N_EMBD];
        rmsnorm(x, N_EMBD, xn2);
        linear(xn2, N_EMBD, off_mlp_fc1[li], MLP_DIM, tmp2);
        for (int j = 0; j < MLP_DIM; j++)
            tmp2[j] = tape_relu(tmp2[j]);
        linear(tmp2, MLP_DIM, off_mlp_fc2[li], N_EMBD, tmp);
        for (int j = 0; j < N_EMBD; j++)
            x[j] = tape_add(tmp[j], x_res[j]);
    }

    /* Final logits via lm_head */
    linear(x, N_EMBD, off_lm_head, vocab_size, logits_out);
}

/* ── Training ──────────────────────────────────────────────────── */

static void train(void) {
    for (int step = 0; step < NUM_STEPS; step++) {
        /* Pick document, tokenize with BOS sentinels */
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

        /* Reset tape & KV cache, place params on tape */
        tape_reset();
        kv_reset();
        params_to_tape();

        /* Forward: accumulate loss over sequence */
        int loss_terms[BLOCK_SIZE];
        for (int pos = 0; pos < n; pos++) {
            int logits[128];
            gpt_forward(tokens[pos], pos, logits);

            int probs[128];
            softmax(logits, vocab_size, probs);

            /* loss_t = -log(prob[target]) */
            int neg1 = tape_leaf(-1.0);
            loss_terms[pos] = tape_mul(tape_log(probs[tokens[pos + 1]]), neg1);
        }

        /* Average loss */
        int total_loss = loss_terms[0];
        for (int i = 1; i < n; i++)
            total_loss = tape_add(total_loss, loss_terms[i]);
        int inv_n = tape_leaf(1.0 / (double)n);
        int loss = tape_mul(total_loss, inv_n);

        /* Backward */
        tape_backward(loss);

        /* Adam update */
        double lr_t = LR * (1.0 - (double)step / NUM_STEPS);
        for (int i = 0; i < num_params; i++) {
            double g = tape[param_tape_idx[i]].grad;
            adam_m[i] = BETA1 * adam_m[i] + (1.0 - BETA1) * g;
            adam_v[i] = BETA2 * adam_v[i] + (1.0 - BETA2) * g * g;
            double m_hat = adam_m[i] / (1.0 - pow(BETA1, step + 1));
            double v_hat = adam_v[i] / (1.0 - pow(BETA2, step + 1));
            param_val[i] -= lr_t * m_hat / (sqrt(v_hat) + EPS_ADAM);
        }

        printf("\rstep %4d / %4d | loss %.4f", step + 1, NUM_STEPS, tape[loss].val);
        fflush(stdout);
    }
    printf("\n");
}

/* ── Inference ─────────────────────────────────────────────────── */

/* Weighted random choice (roulette wheel) */
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
        tape_reset();
        kv_reset();
        params_to_tape();

        int token_id = BOS;
        char name[BLOCK_SIZE + 1];
        int name_len = 0;

        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            int logits[128];
            gpt_forward(token_id, pos, logits);

            /* Apply temperature and softmax (using forward values only) */
            double probs[128];
            double max_l = tape[logits[0]].val;
            for (int i = 1; i < vocab_size; i++)
                if (tape[logits[i]].val > max_l) max_l = tape[logits[i]].val;
            double sum = 0;
            for (int i = 0; i < vocab_size; i++) {
                probs[i] = exp((tape[logits[i]].val / TEMPERATURE) - max_l / TEMPERATURE);
                sum += probs[i];
            }
            for (int i = 0; i < vocab_size; i++) probs[i] /= sum;

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

    /* Try to download if missing */
    FILE *test = fopen(data_path, "r");
    if (!test) {
        printf("Downloading names.txt...\n");
        int ret = system("curl -sfL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt");
        if (ret != 0) {
            /* Try wget as fallback */
            ret = system("wget -q https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -O input.txt");
        }
        if (ret != 0) {
            fprintf(stderr, "Could not download dataset. Place names.txt as input.txt.\n");
            return 1;
        }
    } else {
        fclose(test);
    }

    rng_seed(42);
    load_dataset(data_path);
    init_params();

    clock_t t0 = clock();
    train();
    clock_t t1 = clock();
    printf("Training time: %.2f seconds\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    inference(20);
    return 0;
}
