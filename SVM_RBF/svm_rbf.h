#ifndef __SVM_RBF_H__
#define __SVM_RBF_H__

#include <math.h>

#ifdef USE_CMSIS
#include "arm_math.h"
#else
typedef struct
{
  uint32_t      nbOfSupportVectors;     // 支持向量数目
  uint32_t      vectorDimension;        // 向量维度
  float         intercept;              // 偏移系数
  const float   *dualCoefficients;      // 加权系数
  const float   *supportVectors;        // 支持向量，分nbOfSupportVectors段存放，每段是vectorDimension维的支持向量
  const int32_t *classes;               // 类别编号
  float         gamma;                  // Gamma参数
} arm_svm_rbf_instance_f32;

void arm_svm_rbf_predict_f32(
    const arm_svm_rbf_instance_f32 *S,  // SVM模型数据结构
    const float * in,                   // 输入数组
    int32_t * pResult);                 // 分类结果

#include "mat_f32.h"
#endif

#endif
