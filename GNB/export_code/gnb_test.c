#ifdef USE_CMSIS
#include "arm_math.h"
#else
#include "../gnb.h"
#endif

#include "gnb_test.h"

const float32_t model_theta[NUM_CLS*NUM_DIM] = 
{
    5.006000e+00, 3.428000e+00, 1.462000e+00, 2.460000e-01, 
    5.936000e+00, 2.770000e+00, 4.260000e+00, 1.326000e+00, 
    6.588000e+00, 2.974000e+00, 5.552000e+00, 2.026000e+00, 
};
const float32_t model_sigma[NUM_CLS*NUM_DIM] = 
{
    1.217640e-01, 1.408160e-01, 2.955600e-02, 1.088400e-02, 
    2.611040e-01, 9.650000e-02, 2.164000e-01, 3.832400e-02, 
    3.962560e-01, 1.019240e-01, 2.984960e-01, 7.392400e-02, 
};
const float32_t model_priors[NUM_CLS] = 
{
    3.333333e-01, 3.333333e-01, 3.333333e-01, 
};

const float32_t test_in[NUM_DAT*NUM_DIM] = 
{
    5.100000e+00, 3.500000e+00, 1.400000e+00, 2.000000e-01, 
    4.900000e+00, 3.000000e+00, 1.400000e+00, 2.000000e-01, 
    4.700000e+00, 3.200000e+00, 1.300000e+00, 2.000000e-01, 
    4.600000e+00, 3.100000e+00, 1.500000e+00, 2.000000e-01, 
    5.000000e+00, 3.600000e+00, 1.400000e+00, 2.000000e-01, 
    5.400000e+00, 3.900000e+00, 1.700000e+00, 4.000000e-01, 
    4.600000e+00, 3.400000e+00, 1.400000e+00, 3.000000e-01, 
    5.000000e+00, 3.400000e+00, 1.500000e+00, 2.000000e-01, 
    4.400000e+00, 2.900000e+00, 1.400000e+00, 2.000000e-01, 
    4.900000e+00, 3.100000e+00, 1.500000e+00, 1.000000e-01, 
    5.400000e+00, 3.700000e+00, 1.500000e+00, 2.000000e-01, 
    4.800000e+00, 3.400000e+00, 1.600000e+00, 2.000000e-01, 
    4.800000e+00, 3.000000e+00, 1.400000e+00, 1.000000e-01, 
    4.300000e+00, 3.000000e+00, 1.100000e+00, 1.000000e-01, 
    5.800000e+00, 4.000000e+00, 1.200000e+00, 2.000000e-01, 
    5.700000e+00, 4.400000e+00, 1.500000e+00, 4.000000e-01, 
    5.400000e+00, 3.900000e+00, 1.300000e+00, 4.000000e-01, 
    5.100000e+00, 3.500000e+00, 1.400000e+00, 3.000000e-01, 
    5.700000e+00, 3.800000e+00, 1.700000e+00, 3.000000e-01, 
    5.100000e+00, 3.800000e+00, 1.500000e+00, 3.000000e-01, 
    5.400000e+00, 3.400000e+00, 1.700000e+00, 2.000000e-01, 
    5.100000e+00, 3.700000e+00, 1.500000e+00, 4.000000e-01, 
    4.600000e+00, 3.600000e+00, 1.000000e+00, 2.000000e-01, 
    5.100000e+00, 3.300000e+00, 1.700000e+00, 5.000000e-01, 
    4.800000e+00, 3.400000e+00, 1.900000e+00, 2.000000e-01, 
    5.000000e+00, 3.000000e+00, 1.600000e+00, 2.000000e-01, 
    5.000000e+00, 3.400000e+00, 1.600000e+00, 4.000000e-01, 
    5.200000e+00, 3.500000e+00, 1.500000e+00, 2.000000e-01, 
    5.200000e+00, 3.400000e+00, 1.400000e+00, 2.000000e-01, 
    4.700000e+00, 3.200000e+00, 1.600000e+00, 2.000000e-01, 
    4.800000e+00, 3.100000e+00, 1.600000e+00, 2.000000e-01, 
    5.400000e+00, 3.400000e+00, 1.500000e+00, 4.000000e-01, 
    5.200000e+00, 4.100000e+00, 1.500000e+00, 1.000000e-01, 
    5.500000e+00, 4.200000e+00, 1.400000e+00, 2.000000e-01, 
    4.900000e+00, 3.100000e+00, 1.500000e+00, 2.000000e-01, 
    5.000000e+00, 3.200000e+00, 1.200000e+00, 2.000000e-01, 
    5.500000e+00, 3.500000e+00, 1.300000e+00, 2.000000e-01, 
    4.900000e+00, 3.600000e+00, 1.400000e+00, 1.000000e-01, 
    4.400000e+00, 3.000000e+00, 1.300000e+00, 2.000000e-01, 
    5.100000e+00, 3.400000e+00, 1.500000e+00, 2.000000e-01, 
    5.000000e+00, 3.500000e+00, 1.300000e+00, 3.000000e-01, 
    4.500000e+00, 2.300000e+00, 1.300000e+00, 3.000000e-01, 
    4.400000e+00, 3.200000e+00, 1.300000e+00, 2.000000e-01, 
    5.000000e+00, 3.500000e+00, 1.600000e+00, 6.000000e-01, 
    5.100000e+00, 3.800000e+00, 1.900000e+00, 4.000000e-01, 
    4.800000e+00, 3.000000e+00, 1.400000e+00, 3.000000e-01, 
    5.100000e+00, 3.800000e+00, 1.600000e+00, 2.000000e-01, 
    4.600000e+00, 3.200000e+00, 1.400000e+00, 2.000000e-01, 
    5.300000e+00, 3.700000e+00, 1.500000e+00, 2.000000e-01, 
    5.000000e+00, 3.300000e+00, 1.400000e+00, 2.000000e-01, 
    7.000000e+00, 3.200000e+00, 4.700000e+00, 1.400000e+00, 
    6.400000e+00, 3.200000e+00, 4.500000e+00, 1.500000e+00, 
    6.900000e+00, 3.100000e+00, 4.900000e+00, 1.500000e+00, 
    5.500000e+00, 2.300000e+00, 4.000000e+00, 1.300000e+00, 
    6.500000e+00, 2.800000e+00, 4.600000e+00, 1.500000e+00, 
    5.700000e+00, 2.800000e+00, 4.500000e+00, 1.300000e+00, 
    6.300000e+00, 3.300000e+00, 4.700000e+00, 1.600000e+00, 
    4.900000e+00, 2.400000e+00, 3.300000e+00, 1.000000e+00, 
    6.600000e+00, 2.900000e+00, 4.600000e+00, 1.300000e+00, 
    5.200000e+00, 2.700000e+00, 3.900000e+00, 1.400000e+00, 
    5.000000e+00, 2.000000e+00, 3.500000e+00, 1.000000e+00, 
    5.900000e+00, 3.000000e+00, 4.200000e+00, 1.500000e+00, 
    6.000000e+00, 2.200000e+00, 4.000000e+00, 1.000000e+00, 
    6.100000e+00, 2.900000e+00, 4.700000e+00, 1.400000e+00, 
    5.600000e+00, 2.900000e+00, 3.600000e+00, 1.300000e+00, 
    6.700000e+00, 3.100000e+00, 4.400000e+00, 1.400000e+00, 
    5.600000e+00, 3.000000e+00, 4.500000e+00, 1.500000e+00, 
    5.800000e+00, 2.700000e+00, 4.100000e+00, 1.000000e+00, 
    6.200000e+00, 2.200000e+00, 4.500000e+00, 1.500000e+00, 
    5.600000e+00, 2.500000e+00, 3.900000e+00, 1.100000e+00, 
    5.900000e+00, 3.200000e+00, 4.800000e+00, 1.800000e+00, 
    6.100000e+00, 2.800000e+00, 4.000000e+00, 1.300000e+00, 
    6.300000e+00, 2.500000e+00, 4.900000e+00, 1.500000e+00, 
    6.100000e+00, 2.800000e+00, 4.700000e+00, 1.200000e+00, 
    6.400000e+00, 2.900000e+00, 4.300000e+00, 1.300000e+00, 
    6.600000e+00, 3.000000e+00, 4.400000e+00, 1.400000e+00, 
    6.800000e+00, 2.800000e+00, 4.800000e+00, 1.400000e+00, 
    6.700000e+00, 3.000000e+00, 5.000000e+00, 1.700000e+00, 
    6.000000e+00, 2.900000e+00, 4.500000e+00, 1.500000e+00, 
    5.700000e+00, 2.600000e+00, 3.500000e+00, 1.000000e+00, 
    5.500000e+00, 2.400000e+00, 3.800000e+00, 1.100000e+00, 
    5.500000e+00, 2.400000e+00, 3.700000e+00, 1.000000e+00, 
    5.800000e+00, 2.700000e+00, 3.900000e+00, 1.200000e+00, 
    6.000000e+00, 2.700000e+00, 5.100000e+00, 1.600000e+00, 
    5.400000e+00, 3.000000e+00, 4.500000e+00, 1.500000e+00, 
    6.000000e+00, 3.400000e+00, 4.500000e+00, 1.600000e+00, 
    6.700000e+00, 3.100000e+00, 4.700000e+00, 1.500000e+00, 
    6.300000e+00, 2.300000e+00, 4.400000e+00, 1.300000e+00, 
    5.600000e+00, 3.000000e+00, 4.100000e+00, 1.300000e+00, 
    5.500000e+00, 2.500000e+00, 4.000000e+00, 1.300000e+00, 
    5.500000e+00, 2.600000e+00, 4.400000e+00, 1.200000e+00, 
    6.100000e+00, 3.000000e+00, 4.600000e+00, 1.400000e+00, 
    5.800000e+00, 2.600000e+00, 4.000000e+00, 1.200000e+00, 
    5.000000e+00, 2.300000e+00, 3.300000e+00, 1.000000e+00, 
    5.600000e+00, 2.700000e+00, 4.200000e+00, 1.300000e+00, 
    5.700000e+00, 3.000000e+00, 4.200000e+00, 1.200000e+00, 
    5.700000e+00, 2.900000e+00, 4.200000e+00, 1.300000e+00, 
    6.200000e+00, 2.900000e+00, 4.300000e+00, 1.300000e+00, 
    5.100000e+00, 2.500000e+00, 3.000000e+00, 1.100000e+00, 
    5.700000e+00, 2.800000e+00, 4.100000e+00, 1.300000e+00, 
    6.300000e+00, 3.300000e+00, 6.000000e+00, 2.500000e+00, 
    5.800000e+00, 2.700000e+00, 5.100000e+00, 1.900000e+00, 
    7.100000e+00, 3.000000e+00, 5.900000e+00, 2.100000e+00, 
    6.300000e+00, 2.900000e+00, 5.600000e+00, 1.800000e+00, 
    6.500000e+00, 3.000000e+00, 5.800000e+00, 2.200000e+00, 
    7.600000e+00, 3.000000e+00, 6.600000e+00, 2.100000e+00, 
    4.900000e+00, 2.500000e+00, 4.500000e+00, 1.700000e+00, 
    7.300000e+00, 2.900000e+00, 6.300000e+00, 1.800000e+00, 
    6.700000e+00, 2.500000e+00, 5.800000e+00, 1.800000e+00, 
    7.200000e+00, 3.600000e+00, 6.100000e+00, 2.500000e+00, 
    6.500000e+00, 3.200000e+00, 5.100000e+00, 2.000000e+00, 
    6.400000e+00, 2.700000e+00, 5.300000e+00, 1.900000e+00, 
    6.800000e+00, 3.000000e+00, 5.500000e+00, 2.100000e+00, 
    5.700000e+00, 2.500000e+00, 5.000000e+00, 2.000000e+00, 
    5.800000e+00, 2.800000e+00, 5.100000e+00, 2.400000e+00, 
    6.400000e+00, 3.200000e+00, 5.300000e+00, 2.300000e+00, 
    6.500000e+00, 3.000000e+00, 5.500000e+00, 1.800000e+00, 
    7.700000e+00, 3.800000e+00, 6.700000e+00, 2.200000e+00, 
    7.700000e+00, 2.600000e+00, 6.900000e+00, 2.300000e+00, 
    6.000000e+00, 2.200000e+00, 5.000000e+00, 1.500000e+00, 
    6.900000e+00, 3.200000e+00, 5.700000e+00, 2.300000e+00, 
    5.600000e+00, 2.800000e+00, 4.900000e+00, 2.000000e+00, 
    7.700000e+00, 2.800000e+00, 6.700000e+00, 2.000000e+00, 
    6.300000e+00, 2.700000e+00, 4.900000e+00, 1.800000e+00, 
    6.700000e+00, 3.300000e+00, 5.700000e+00, 2.100000e+00, 
    7.200000e+00, 3.200000e+00, 6.000000e+00, 1.800000e+00, 
    6.200000e+00, 2.800000e+00, 4.800000e+00, 1.800000e+00, 
    6.100000e+00, 3.000000e+00, 4.900000e+00, 1.800000e+00, 
    6.400000e+00, 2.800000e+00, 5.600000e+00, 2.100000e+00, 
    7.200000e+00, 3.000000e+00, 5.800000e+00, 1.600000e+00, 
    7.400000e+00, 2.800000e+00, 6.100000e+00, 1.900000e+00, 
    7.900000e+00, 3.800000e+00, 6.400000e+00, 2.000000e+00, 
    6.400000e+00, 2.800000e+00, 5.600000e+00, 2.200000e+00, 
    6.300000e+00, 2.800000e+00, 5.100000e+00, 1.500000e+00, 
    6.100000e+00, 2.600000e+00, 5.600000e+00, 1.400000e+00, 
    7.700000e+00, 3.000000e+00, 6.100000e+00, 2.300000e+00, 
    6.300000e+00, 3.400000e+00, 5.600000e+00, 2.400000e+00, 
    6.400000e+00, 3.100000e+00, 5.500000e+00, 1.800000e+00, 
    6.000000e+00, 3.000000e+00, 4.800000e+00, 1.800000e+00, 
    6.900000e+00, 3.100000e+00, 5.400000e+00, 2.100000e+00, 
    6.700000e+00, 3.100000e+00, 5.600000e+00, 2.400000e+00, 
    6.900000e+00, 3.100000e+00, 5.100000e+00, 2.300000e+00, 
    5.800000e+00, 2.700000e+00, 5.100000e+00, 1.900000e+00, 
    6.800000e+00, 3.200000e+00, 5.900000e+00, 2.300000e+00, 
    6.700000e+00, 3.300000e+00, 5.700000e+00, 2.500000e+00, 
    6.700000e+00, 3.000000e+00, 5.200000e+00, 2.300000e+00, 
    6.300000e+00, 2.500000e+00, 5.000000e+00, 1.900000e+00, 
    6.500000e+00, 3.000000e+00, 5.200000e+00, 2.000000e+00, 
    6.200000e+00, 3.400000e+00, 5.400000e+00, 2.300000e+00, 
    5.900000e+00, 3.000000e+00, 5.100000e+00, 1.800000e+00, 
};
const uint32_t test_out[] = 
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    2, 2, 2, 2, 2, 2, 
};

int32_t gnb_test()
{
    uint32_t err=0;
    float32_t prob[NUM_CLS];

    arm_gaussian_naive_bayes_instance_f32 model;
    model.vectorDimension=NUM_DIM;
    model.numberOfClasses=NUM_CLS;
    model.theta=model_theta;
    model.sigma=model_sigma;
    model.classPriors=model_priors;
    model.epsilon=0.0;

    err=0;
    for (int n=0; n<NUM_DAT; n++)
        if (test_out[n]!=arm_gaussian_naive_bayes_predict_f32(&model,test_in+n*NUM_DIM,prob))
            err++;
    return err;
}

#include <stdio.h>
int main()
{
    int32_t num_err=gnb_test();
    printf("[INF] test error: %d/%d\n",num_err,NUM_DAT);
}