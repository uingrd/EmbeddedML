#ifndef __GNB_H__
#define __GNB_H__

#include <math.h>
#include <stdint.h>

#ifndef PI_F
#define PI_F 3.1415926535897932384626433832795f
#endif

#ifndef DPI_F
#define DPI_F (2.0f*3.1415926535897932384626433832795f)
#endif

#ifdef USE_CMSIS
#include "arm_math.h"
#else

#ifndef float32_t
typedef float float32_t;
#endif

void arm_max_f32(float32_t *p, uint32_t size, float32_t *max, uint32_t *index);

typedef struct
{
  uint32_t vectorDimension;     /**< Dimension of vector space */
  uint32_t numberOfClasses;     /**< Number of different classes  */
  const float32_t *theta;       /**< Mean values for the Gaussians */
  const float32_t *sigma;       /**< Variances for the Gaussians */
  const float32_t *classPriors; /**< Class prior probabilities */
  float32_t epsilon;            /**< Additive value to variances */
} arm_gaussian_naive_bayes_instance_f32;

uint32_t arm_gaussian_naive_bayes_predict_f32(
    const arm_gaussian_naive_bayes_instance_f32 *S,
    const float32_t * in, 
    float32_t *pBuffer);
#endif

#endif
