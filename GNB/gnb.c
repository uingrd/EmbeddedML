#include <math.h>
#include <stdint.h>

#ifdef USE_CMSIS
#include "arm_math.h"
#else

#include "gnb.h"

void arm_max_f32(float32_t *p, uint32_t size, float32_t *max, uint32_t *index)
{
    *max=p[0];
    *index=0;
    for(uint32_t n=1; n<size; n++)
    {
        if (p[n]>*max)
        {
            *max=p[n];
            *index=n;
        }
    }
    
}

//                    p(y)*p(x1|y)p(x2|y)...
// p(y|x1,x2,...)=  --------------------------
//                        p(x1)p(x2)...
// 这里计算概率的对数
// S->numberOfClasses   y的离散取值个数
// S->vectorDimension   x观测量个数
// S->classPriors       先验p(y)
// S->theta和S->sigma   分别是每个Y的取值和每个x观测量对应的高斯模型参数
//                      分段存放，每个y的取值一段(每段长x的个数）
// in                   观测向量，就是x1, x2, ...
// pBuffer              存放y的每个类别的概率对数
uint32_t arm_gaussian_naive_bayes_predict_f32(
    const arm_gaussian_naive_bayes_instance_f32 *S, // 数据结构
    const float32_t * in, 
    float32_t *pBuffer)
{
    uint32_t nbClass;
    uint32_t nbDim;
    const float32_t *pPrior = S->classPriors;
    const float32_t *pTheta = S->theta;
    const float32_t *pSigma = S->sigma;
    float32_t *buffer = pBuffer;
    const float32_t *pIn=in;
    float32_t result;
    float32_t sigma;
    float32_t tmp;
    float32_t acc1,acc2;
    uint32_t index;

    pTheta=S->theta;
    pSigma=S->sigma;

    for(nbClass = 0; nbClass < S->numberOfClasses; nbClass++)   // 遍历每个Y的类别
    {
        pIn  = in;          // 恢复pIn指针（指向各个观测量）
        tmp  = 0.0;
        acc1 = 0.0f;
        acc2 = 0.0f;
        for(nbDim = 0; nbDim < S->vectorDimension; nbDim++)     // 遍历每个观测量（假设每个是独立高斯）
        {
           sigma = *pSigma + S->epsilon;
           acc1 += logf(2.0f * PI_F * sigma);                   // 对数相加用于实现相乘
           acc2 += (*pIn - *pTheta) * (*pIn - *pTheta) / sigma; // exp相乘对应指数相加

           pIn++;
           pTheta++;        // 每个观测量序号和每个Y的离散取值（类别）对应一个模型
           pSigma++;        // 同上
        }

        tmp = -0.5f * acc1; // 作用在对数上，对应开根号倒数
        tmp -= 0.5f * acc2; // 减号是因为exp指数上是负的，前面计算acc2的时候没有取负号，这里补上

        *buffer = tmp + logf(*pPrior++);                        // 对数相加对应乘以先验
        buffer++;
    }

    arm_max_f32(pBuffer,S->numberOfClasses,&result,&index);

    return(index);
}

#endif

