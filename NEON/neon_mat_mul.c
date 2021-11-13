#include <arm_neon.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void mat_mul(float32_t *A, float32_t *B, float32_t *C,\
            uint32_t n, uint32_t m, uint32_t k) 
{
    for (int i_idx=0; i_idx < n; i_idx++) 
    {
        for (int j_idx=0; j_idx < m; j_idx++) 
        {
            C[n*j_idx + i_idx] = 0;
            for (int k_idx=0; k_idx < k; k_idx++) 
            {
                C[n*j_idx + i_idx] += A[n*k_idx + i_idx]*\
                                      B[k*j_idx + k_idx];
            }
        }
    }
}

// 代码清单8-33
void mat_mul_blk4(float32_t  *A, float32_t  *B, float32_t *C, \
              uint32_t n, uint32_t m, uint32_t k) 
{
	int A_idx, B_idx, C_idx;
	
	float32x4_t A0, A1, A2, A3;
	float32x4_t B0, B1, B2, B3;
	float32x4_t C0, C1, C2, C3;
	
	for (int i_idx=0; i_idx<n; i_idx+=4) 
    {
        for (int j_idx=0; j_idx<m; j_idx+=4)
        {
            // zero accumulators before matrix op
            C0=vmovq_n_f32(0);
            C1=vmovq_n_f32(0);
            C2=vmovq_n_f32(0); 
            C3=vmovq_n_f32(0);
            
            for (int k_idx=0; k_idx<k; k_idx+=4)
            {
                // compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx+k_idx;

                // 加载A的4x4子矩阵的4列
                A0=vld1q_f32(A+A_idx    );
                A1=vld1q_f32(A+A_idx+n  );
                A2=vld1q_f32(A+A_idx+2*n);
                A3=vld1q_f32(A+A_idx+3*n);

                // 加载B的4x4子矩阵第0列
                B0=vld1q_f32(B+B_idx);
                // 将4个元素分别和A子矩阵的4个列相乘，
                // 结果相加后存放4元素列向量C0
                C0=vmlaq_n_f32(C0, A0, B0[0]);
                C0=vmlaq_n_f32(C0, A1, B0[1]);
                C0=vmlaq_n_f32(C0, A2, B0[2]);
                C0=vmlaq_n_f32(C0, A3, B0[3]);
                
                // 加载B的4x4子矩阵第1列
                B1=vld1q_f32(B+B_idx+k);
                // 将4个元素分别和A子矩阵的4个列相乘，
                // 结果相加后存放4元素列向量C1
                C1=vmlaq_n_f32(C1,A0,B1[0]);
                C1=vmlaq_n_f32(C1,A1,B1[1]);
                C1=vmlaq_n_f32(C1,A2,B1[2]);
                C1=vmlaq_n_f32(C1,A3,B1[3]);

                // 加载B的4x4子矩阵第2列
                B2=vld1q_f32(B+B_idx+2*k);
                // 将4个元素分别和A子矩阵的4个列相乘，
                // 结果相加后存放4元素列向量C2
                C2=vmlaq_n_f32(C2,A0,B2[0]);
                C2=vmlaq_n_f32(C2,A1,B2[1]);
                C2=vmlaq_n_f32(C2,A2,B2[2]);
                C2=vmlaq_n_f32(C2,A3,B3[3]);

                // 加载B的4x4子矩阵第3列
                B3=vld1q_f32(B+B_idx+3*k);
                // 将4个元素分别和A子矩阵的4个列相乘，
                // 结果相加后存放4元素列向量C3
                C3=vmlaq_n_f32(C3,A0,B3[0]);
                C3=vmlaq_n_f32(C3,A1,B3[1]);
                C3=vmlaq_n_f32(C3,A2,B3[2]);
                C3=vmlaq_n_f32(C3,A3,B3[3]);
            }
            
            // 分块乘积结果保存C矩阵对应位置
            C_idx = n*j_idx + i_idx;
            vst1q_f32(C+C_idx    ,C0);
            vst1q_f32(C+C_idx+n  ,C1);
            vst1q_f32(C+C_idx+2*n,C2);
            vst1q_f32(C+C_idx+3*n,C3);
        }
    }
}

int main()
{
    int n=12, m=16, k=8;
    float32_t err=0;
    float32_t *A=malloc(sizeof(float32_t)*n*k);
    float32_t *B=malloc(sizeof(float32_t)*m*k);
    float32_t *C=malloc(sizeof(float32_t)*n*m);
    float32_t *D=malloc(sizeof(float32_t)*n*m);
    
    for (int c=0; c<n*k; c++) A[c]=(float32_t)rand()/(float32_t)RAND_MAX;
    for (int c=0; c<m*k; c++) B[c]=(float32_t)rand()/(float32_t)RAND_MAX;
    for (int c=0; c<n*m; c++) C[c]=D[c]=(float32_t)rand()/(float32_t)RAND_MAX;
    
    printf("mat_mul\n");
    mat_mul(A,B,C,n,m,k);
    
    printf("mat_mul_blk4\n");
    mat_mul_blk4(A,B,D,n,m,k);
    
    printf("calc error...\n");
    for (int c=0; c<n*m; c++)
        err+=fabs(C[c]-D[c]);
    printf("    %f\n",err);
    
    free(A);
    free(B);
    free(C);
    return 0;
}
