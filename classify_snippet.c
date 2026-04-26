/*
 * classify_snippet.c — AVR 1‑NN classifier in LDA space (compressed int16 version)
 */

#ifndef F_CPU
#  define F_CPU 11059200UL
#endif

#include <avr/pgmspace.h>
#include <math.h>
#include <stdint.h>

#include "word_templates.h"

uint8_t classify_word(float fv[N_FEATURES])
{
    float y[LDA_DIMS];
    uint8_t d;

    for (d = 0; d < LDA_DIMS; d++) {
        y[d] = 0.0f;
    }

    uint16_t i;
    for (i = 0; i < N_FEATURES; i++) {
        int16_t mean_i16 = (int16_t)pgm_read_word(&feature_mean[i]);
        int16_t std_i16  = (int16_t)pgm_read_word(&feature_std[i]);
        float mean_i = (float)mean_i16 / MODEL_SCALE;
        float std_i  = (float)std_i16  / MODEL_SCALE;
        if (std_i < 1e-9f) std_i = 1e-9f;

        float z_i = (fv[i] - mean_i) / std_i;
        if (!(z_i <= 1.0e30f && z_i >= -1.0e30f)) {
            z_i = 0.0f;
        }

        for (d = 0; d < LDA_DIMS; d++) {
            int16_t w_i16 = (int16_t)pgm_read_word(&lda_W[i][d]);
            float w = (float)w_i16 / MODEL_SCALE;
            y[d] += z_i * w;
        }
    }

    for (d = 0; d < LDA_DIMS; d++) {
        int16_t xb_i16 = (int16_t)pgm_read_word(&lda_xbar[d]);
        y[d] -= (float)xb_i16 / MODEL_SCALE;
    }

    float best_dist_sq = 1.0e20f;
    uint8_t best_word = 0xFF;
    uint8_t w, k;

    for (w = 0; w < N_WORDS; w++) {
        for (k = 0; k < K_TEMPLATES; k++) {
            float dist_sq = 0.0f;
            for (d = 0; d < LDA_DIMS; d++) {
                int16_t t_i16 = (int16_t)pgm_read_word(&lda_templates[w][k][d]);
                float t_val = (float)t_i16 / MODEL_SCALE;
                float diff = y[d] - t_val;
                dist_sq += diff * diff;
            }
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_word = w;
            }
        }
    }

    if (best_dist_sq > (CONFIDENCE_THRESHOLD * CONFIDENCE_THRESHOLD)) {
        return 0xFF;
    }
    return best_word;
}