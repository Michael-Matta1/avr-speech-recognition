/*
 * classify_snippet.c
 *
 * Reference AVR inference routine for ATmega32A deployment.
 * Include this function in your main firmware source and call it with the
 * extracted feature vector defined by the training pipeline.
 *
 * Pipeline:
 *   1. Z-score normalize fv[N_FEATURES]
 *   2. Project to LDA space: proj[LDA_DIMS] = fv @ lda_W - lda_xbar
 *   3. 1-NN over all words x templates in LDA space
 *      (optionally class-scaled if lda_distance_scale is exported)
 *   4. Reject if best distance > CONFIDENCE_THRESHOLD
 */
#include "word_templates.h"

uint8_t classify_word(float fv[N_FEATURES]) {
    uint8_t i, d, w, k;

    /* 1. Z-score normalize */
    for (i = 0; i < N_FEATURES; i++) {
        float m = pgm_read_float(&feature_mean[i]);
        float s = pgm_read_float(&feature_std[i]);
        fv[i] = (fv[i] - m) / s;
    }

    /* 2. LDA projection: proj = fv @ lda_W - lda_xbar */
    float proj[LDA_DIMS];
    for (d = 0; d < LDA_DIMS; d++) {
        proj[d] = 0.0f;
        for (i = 0; i < N_FEATURES; i++) {
            proj[d] += fv[i] * pgm_read_float(&lda_W[i][d]);
        }
        proj[d] -= pgm_read_float(&lda_xbar[d]);
    }

    /* 3. 1-NN over all words x K_TEMPLATES templates */
    float   best_dist = 1e9f;
    uint8_t best_word = 0xFF;   /* 0xFF = unknown */

    for (w = 0; w < N_WORDS; w++) {
        for (k = 0; k < K_TEMPLATES; k++) {
            float dist = 0.0f;
            for (d = 0; d < LDA_DIMS; d++) {
                float diff = proj[d]
                           - pgm_read_float(&lda_templates[w][k][d]);
#if defined(LDA_DISTANCE_SCALE_AVAILABLE) && (LDA_DISTANCE_SCALE_AVAILABLE == 1)
                diff *= pgm_read_float(&lda_distance_scale[w][d]);
#endif
                dist += diff * diff;
            }
            dist = sqrtf(dist);
            if (dist < best_dist) {
                best_dist = dist;
                best_word = w;
            }
        }
    }

    if (best_dist > CONFIDENCE_THRESHOLD) return 0xFF;  /* reject */
    return best_word;
}
