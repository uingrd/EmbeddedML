#include <math.h>
#include <stdint.h>

#ifdef USE_CMSIS
#include "arm_math.h"
#else
#include "svm_rbf.h"

void arm_svm_rbf_predict_f32(
    const arm_svm_rbf_instance_f32 *S,  // SVM模型数据结构
    const float * in,                   // 输入数组
    int32_t * pResult)                  // 分类结果
{
    float sum=S->intercept;             // 平移量预先放置到加权和变量
    float dot=0;
    uint32_t i,j;
    const float *pSupport = S->supportVectors;

    for(i=0; i < S->nbOfSupportVectors; i++)
    {
        // 计算输入向量到支持向量的距离
        dot=0;
        for(j=0; j < S->vectorDimension; j++)   // 求各个维度坐标平方和(为了计算向量距离)   
        {
            dot = dot + (in[j]-*pSupport)*(in[j]-*pSupport);
            pSupport++;                         // 指向支持向量的各个元素
        }
        sum += S->dualCoefficients[i] * expf(-S->gamma * dot);  // 距离
    }
    *pResult=S->classes[sum>0];
}
#endif
