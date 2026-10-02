/* Abliteration / activation steering per il residual stream di GLM-5.2.
 *
 * Modo STEER (inference): carica un file di direzioni (una per layer,
 * D float ciascuna) e proietta ogni residual post-layer lungo la direzione
 * del rifiuto:  x' = x - c * (x . r) / (r . r) * r
 *
 * Modo COLLECT (raccolta attivazioni): scrive il residual post-layer su disco
 * per ogni token dell'ultima posizione del prompt, per eseguire poi PCA offline
 * e ottenere le direzioni r_l.
 *
 * Formato file direzioni (.bin, raw little-endian float32):
 *   [n_layers][D] float32  ->  steer_r[layer*D + d]
 *   n_layers = num_hidden_layers (75), D = hidden_size.
 *
 * Env vars:
 *   COLI_STEER_FILE=path   file direzioni da caricare (mmap)
 *   COLI_STEER_COEF=float  coefficiente di proiezione (default 1.0)
 *   COLI_STEER_LAYERS=a-b  applica solo ai layer in [a,b] (default tutti)
 *   COLI_COLLECT_DIR=dir   modalita' raccolta: dump residual per ogni layer qui
 */
#ifndef STEER_H
#define STEER_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>
#include <errno.h>

static float *g_steer_r = NULL;      /* mmap del file direzioni [n_layers * D] */
static int    g_steer_n  = 0;        /* numero layer con direzioni caricate */
static int    g_steer_D  = 0;        /* dimensione hidden */
static double g_steer_coef = 1.0;
static int    g_steer_lo = 0, g_steer_hi = -1;   /* range layer (hi=-1 = tutti) */
static int    g_steer_on = 0;        /* steering attivo */
static const char *g_collect_dir = NULL;          /* modalita' raccolta */
static FILE  *g_collect_f = NULL;
static int    g_collect_wrote = 0;   /* rounds written this prompt (0 = first/prefill) */
static int    g_collect_maxrounds = 1; // COLI_COLLECT_ROUNDS, default 1 (prefill only)

/* Inizializza da env. Chiamare dopo cfg_load (serve D e n_layers). */
static void steer_init(int D, int n_layers){
    g_steer_n = n_layers;   /* needed by steer_collect round-gate even in collect mode */
    const char *fp = getenv("COLI_STEER_FILE");
    const char *cc = getenv("COLI_STEER_COEF");
    const char *lr = getenv("COLI_STEER_LAYERS");
    g_collect_dir = getenv("COLI_COLLECT_DIR");
    { const char *cr=getenv("COLI_COLLECT_ROUNDS"); if(cr) g_collect_maxrounds=atoi(cr); }
    g_steer_D = D;

    /* modalita' COLLECT: apri il file di dump (una riga per layer per token) */
    if(g_collect_dir){
        char path[2048];
        /* ensure the collect dir exists (mkdir -p) so fopen(wb) doesn't fail */
        char mk[2048];
        snprintf(mk, sizeof(mk), "mkdir -p \"%s\"", g_collect_dir);
        int rc = system(mk);
        (void)rc;
        snprintf(path, sizeof(path), "%s/residuals.bin", g_collect_dir);
        g_collect_f = fopen(path, "wb");
        if(!g_collect_f){ fprintf(stderr, "[steer] cannot open collect file %s\n", path); return; }
        /* header: [n_layers][D] cosi' il reader sa la shape */
        int32_t hdr[2] = { n_layers, D };
        fwrite(hdr, sizeof(int32_t), 2, g_collect_f);
        fprintf(stderr, "[steer] COLLECT mode -> %s (n_layers=%d D=%d)\n", path, n_layers, D);
        return;
    }

    /* modalita' STEER: mmap il file direzioni */
    if(!fp) return;
    int fd = open(fp, O_RDONLY);
    if(fd < 0){ fprintf(stderr, "[steer] cannot open %s: %s\n", fp, strerror(errno)); return; }
    struct stat st;
    if(fstat(fd, &st) < 0){ fprintf(stderr, "[steer] fstat fail\n"); close(fd); return; }
    size_t need = (size_t)n_layers * (size_t)D * sizeof(float);
    if((size_t)st.st_size < need){
        fprintf(stderr, "[steer] file %s too small: %zu < %zu\n", fp, (size_t)st.st_size, need);
        close(fd); return;
    }
    void *m = mmap(NULL, need, PROT_READ, MAP_PRIVATE, fd, 0);
    if(m == MAP_FAILED){ fprintf(stderr, "[steer] mmap fail: %s\n", strerror(errno)); close(fd); return; }
    g_steer_r = (float*)m;
    g_steer_n = n_layers;
    g_steer_on = 1;
    if(cc) g_steer_coef = atof(cc);
    if(lr){ int a,b; if(sscanf(lr,"%d-%d",&a,&b)==2){ g_steer_lo=a; g_steer_hi=b; } else g_steer_lo=atoi(lr); }
    if(g_steer_hi < 0) g_steer_hi = n_layers - 1;
    fprintf(stderr, "[steer] STEER on: file=%s coef=%.3f layers=%d-%d D=%d n=%d\n",
            fp, g_steer_coef, g_steer_lo, g_steer_hi, D, n_layers);
}

/* Proietta il residual stream x[0..S*D] lungo la direzione del layer `layer`.
 * Chiamare DOPO il layer_forward, PRIMA del layer successivo. */
static void steer_apply(float *x, int S, int D, int layer){
    if(!g_steer_on) return;
    if(layer < g_steer_lo || layer > g_steer_hi) return;
    if(layer >= g_steer_n) return;
    const float *r = g_steer_r + (size_t)layer * D;
    /* norma della direzione (precomputabile, ma D e' piccolo ~7168 -> ok per token) */
    double rr = 0.0;
    for(int d=0; d<D; d++) rr += (double)r[d] * r[d];
    if(rr < 1e-12) return;   /* direzione nulla: salta */
    double inv_rr = 1.0 / rr;
    for(int s=0; s<S; s++){
        float *row = x + (size_t)s * D;
        double dot = 0.0;
        for(int d=0; d<D; d++) dot += (double)row[d] * r[d];
        double k = g_steer_coef * dot * inv_rr;
        float kf = (float)k;
        for(int d=0; d<D; d++) row[d] -= kf * r[d];
    }
}

/* Modalita' COLLECT: dump del residual del layer corrente (ultima posizione del batch).
 * Chiamare dopo layer_forward, uguale a steer_apply. Scrive D float per ogni layer. */
static void steer_collect(float *x, int S, int D, int layer){
    if(!g_collect_f) return;
    if(g_collect_wrote >= g_collect_maxrounds) return;   /* only first round(s): prefill */
    const float *row = x + (size_t)(S-1) * D;
    fwrite(row, sizeof(float), D, g_collect_f);
    if(layer == g_steer_n-1) g_collect_wrote++;
}

static void steer_close(void){
    if(g_collect_f){ fclose(g_collect_f); g_collect_f=NULL; }
    if(g_steer_r){ munmap(g_steer_r, (size_t)g_steer_n * g_steer_D * sizeof(float)); g_steer_r=NULL; }
}
#endif
