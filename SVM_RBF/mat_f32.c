#include <stdint.h>
#include "mat_f32.h"
void mat_init_f32(mat_f32_s *pmat, uint32_t num_row, uint32_t num_col, float *pdata)
{
    pmat->num_row=num_row;
    pmat->num_col=num_col;
    pmat->pdata=pdata;
}

// FIXME! not size check
void mat_add_f32(mat_f32_s *in1, mat_f32_s *in2, mat_f32_s *out)
{   
    uint32_t size=in1->num_col*in1->num_row;
    for (uint32_t n=0; n<size; n++) 
        out->pdata[n]=in1->pdata[n]+in2->pdata[n];
}

void mat_sub_f32(mat_f32_s *in1, mat_f32_s *in2, mat_f32_s *out)
{   
    uint32_t size=in1->num_col*in1->num_row;
    for (uint32_t n=0; n<size; n++) 
        out->pdata[n]=in1->pdata[n]-in2->pdata[n];
}

void mat_mul_f32(mat_f32_s *in1, mat_f32_s *in2, mat_f32_s *out)
{   
    for (uint32_t n=0; n<in1->num_row; n++) 
        for (uint32_t m=0; m<in2->num_col; m++)
        {
            MAT_ELEM(out,n,0)=0;
            for (uint32_t k=0; k<in1->num_col; k++)
                MAT_ELEM(out,n,m)+=MAT_ELEM(in1,n,k)*MAT_ELEM(in2,k,m);
        }
}
