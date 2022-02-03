#ifndef __MAT_F32_H__
#define __MAT_F32_H__

typedef struct
{
    uint32_t num_row;
    uint32_t num_col;
    float *pdata;
} mat_f32_s;

void mat_init_f32(mat_f32_s *pmat, uint32_t num_row, uint32_t num_col, float *pdata);
void mat_add_f32(mat_f32_s *in1, mat_f32_s *in2, mat_f32_s *out);
void mat_sub_f32(mat_f32_s *in1, mat_f32_s *in2, mat_f32_s *out);
void mat_mul_f32(mat_f32_s *in1, mat_f32_s *in2, mat_f32_s *out);

#define MAT_ELEM(m,r,c) ((m)->pdata[(m)->num_col*(r)+(c)])
#define arm_matrix_instance_f32 mat_f32_s
#define arm_mat_init_f32(p,r,c,buf) do {((mat_f32_s*)(p))->pdata=(buf); ((mat_f32_s*)(p))->num_row=(r); ((mat_f32_s*)(p))->num_col=(c); } while(0)
#define arm_mat_add_f32(p_in1, p_in2, p_out) mat_add_f32(((mat_f32_s*)(p_in1)),((mat_f32_s*)(p_in2)),((mat_f32_s*)(p_out)))
#define arm_mat_sub_f32(p_in1, p_in2, p_out) mat_sub_f32(((mat_f32_s*)(p_in1)),((mat_f32_s*)(p_in2)),((mat_f32_s*)(p_out)))
#define arm_mat_mult_f32(p_in1, p_in2, p_out) mat_mul_f32(((mat_f32_s*)(p_in1)),((mat_f32_s*)(p_in2)),((mat_f32_s*)(p_out)))

#endif
