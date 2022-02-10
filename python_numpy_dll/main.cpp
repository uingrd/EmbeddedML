
#include <stdio.h>

extern  "C"
{  
    int add(int a, int b);
    float add_f32(float a, float b);
    int add_point(int *buf);
    void add_point_io(int *buf_in, int *buf_out);
    float sum_f32_point(float *buf, long sz);
    void add_f32_point(float *buf_in1, float *buf_in2, float *buf_out, long sz);
    void cumsum_f32_point(double *buf_in, double *buf_out, long sz);
} 


int add(int a, int b) { return a+b; }

float add_f32(float a, float b) { return a+b; }

int add_point(int *buf) { return buf[0]+buf[1]; }

void add_point_io(int *buf_in, int *buf_out) 
{ 
    buf_out[0]=buf_in[0]+buf_in[1]; 
    buf_out[1]=buf_in[0]-buf_in[1]; 
}

float sum_f32_point(float *buf, long sz)
{
    float res=0;
    for (long n=0;n<sz;n++) res+=buf[n];
    return res; 
}

void add_f32_point(float *buf_in1, float *buf_in2, float *buf_out, long sz)
{
    for (long n=0;n<sz;n++) buf_out[n]=buf_in1[n]+buf_in2[n];
}   

void cumsum_f32_point(double *buf_in, double *buf_out, long sz)
{
    buf_out[0]=buf_in[0];
    for (long n=1;n<sz;n++)
        buf_out[n]=buf_out[n-1]+buf_in[n]; 
}

