
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// 通过指数字段操作，计算浮点数x*pow(2,s)
float float_shift(float x, int s)
{
    unsigned long* y = (unsigned long*)&x;
    *y += s << 23; // 对应单精度浮点数指数域：23~30-bit 
    return *(float*)y;
}


// 局部序列FIR滤波运算

// 2-tap, output 2
void fir_2_2(float x0, float x1, float x2, float h0, float h1, float *y)
{
    float m1=(x0-x1)*h1;
    float m2=x1*(h0+h1);
    float m3=(x1-x2)*h0;

    y[0]=m1+m2;
    y[1]=m2-m3;
    
    return;
 }   


// 3-tap, output 2
void fir_2_3(float x0, float x1, float x2, float x3, float h0, float h1, float h2, float *y)
{
    float m1=(x0-x2)*h2;
    float m2=float_shift((x1+x2)*(h0+h1+h2),-1);
    float m3=float_shift((x2-x1)*(h0-h1+h2),-1);
    float m4=(x1-x3)*h0;

    y[0]=m1+m2+m3;
    y[1]=m2-m3-m4;
    
    return;
}

// 4-tap, output 4
void fir_4_4(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float h0, float h1, float h2, float h3, float *y)
{
    float M1[2],M2[2],M3[2];
    fir_2_2(x0-x2,x1-x3,x2-x4,h2   ,h3   ,M1);  
    fir_2_2(x2   ,x3   ,x4   ,h0+h2,h1+h3,M2);
    fir_2_2(x2-x4,x3-x5,x4-x6,h0   ,h1   ,M3);
    
    y[0]=M1[0]+M2[0];
    y[1]=M1[1]+M2[1];
    y[2]=M2[0]-M3[0];
    y[3]=M2[1]-M3[1];
    
    return;
}


// 4-tap, output 2
void fir_2_4(float x0, float x1, float x2, float x3, float x4, float h0, float h1, float h2, float h3, float *y)
{
    float M0[2],M1[2];
    fir_2_2(x0,x1,x2,h2,h3,M0);
    fir_2_2(x2,x3,x4,h0,h1,M1);
    y[0]=M0[0]+M1[0];
    y[1]=M0[1]+M1[1];
    return;
}

// 5-tap, output 2
void fir_2_5(float x0, float x1, float x2, float x3, float x4, float x5, float h0, float h1, float h2, float h3, float h4, float *y)
{
    float M0[2],M1[2];
    fir_2_2(x0,x1,x2,h3,h4,M0);
    fir_2_3(x2,x3,x4,x5,h0,h1,h2,M1);
    y[0]=M0[0]+M1[0];
    y[1]=M0[1]+M1[1];
    return;
}

// 连续序列FIR滤波

// 2-tap
void fir_2_2_seq(float *x, int len_x, float *h, float *y)
{
    for (int n=0; n<len_x-2; n+=2) 
        fir_2_2(x[n],x[n+1],x[n+2],h[0],h[1],y+n);
}

// 3-tap
void fir_2_3_seq(float *x, int len_x, float *h, float *y)
{
    for (int n=0; n<len_x-3; n+=2) 
        fir_2_3(x[n],x[n+1],x[n+2],x[n+3],h[0],h[1],h[2],y+n);
}

// 4-tap
void fir_4_4_seq(float *x, int len_x, float *h, float *y)
{
    for (int n=0; n<len_x-6; n+=4) 
        fir_4_4(x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],x[n+6],h[0],h[1],h[2],h[3],y+n);
}

// 4-tap
void fir_2_4_seq(float *x, int len_x, float *h, float *y)
{
    for (int n=0; n<len_x-4; n+=2) 
        fir_2_4(x[n],x[n+1],x[n+2],x[n+3],x[n+4],h[0],h[1],h[2],h[3],y+n);
}

// 5-tap
void fir_2_5_seq(float *x, int len_x, float *h, float *y)
{
    for (int n=0; n<len_x-5; n+=2) 
        fir_2_5(x[n],x[n+1],x[n+2],x[n+3],x[n+4],x[n+5],h[0],h[1],h[2],h[3],h[4],y+n);
}    

// 单元测试代码

// direct 1D conv.
void conv(float *x, int len_x, float *h, int len_h, float *y)
{
    for (int n=0; n<len_x+len_h-1; n++)
    {
        y[n]=0;
        for (int m=0; m<len_h; m++)
            if ((n>=m)&&(n-m<len_x))
                y[n]+=x[n-m]*h[m];
    }
    
    return;
}

float randu(float min, float max)
{
    float f=((float)random())/((float)RAND_MAX);
    return f*(max-min)+min;
}


float calc_err(float *a, float *b, int len)
{
    float e=0;
    for (int n=0; n<len; n++)
        e+=fabs(a[n]-b[n])/fabs(b[n]);
    return e/((float)len);
}

#define N 5000   // 滤波序列长度
int  main()
{
    float x[N];
    float h[5];
    float y[N+5-1];
    float y_ref[N+5-1];
    
    // 测试数据生成
    for (int n=0;n<N;n++) x[n]=randu(-100.0,100.0);
    for (int n=0;n<5;n++) h[n]=randu(-100.0,100.0);
    
    // FIR滤波器测试
    printf("[INF] ==== testing 2-tap (output 2) FIR fir_2_2()...\n");
    conv(x, N, h, 2, y_ref);    // 参考答案
    fir_2_2_seq(x,N,h,y);       // 快速算法
    printf("[INF] error:%.4e\n",calc_err(y,y_ref+1,N-2));
    
    printf("[INF] ==== testing 3-tap (output 2) FIR fir_2_3()...\n");
    conv(x, N, h, 3, y_ref);    // 参考答案
    fir_2_3_seq(x,N,h,y);       // 快速算法
    printf("[INF] error:%.4e\n",calc_err(y,y_ref+2,N-2));

    printf("[INF] ==== testing 4-tap (output 4) FIR fir_4_4()...\n");
    conv(x, N, h, 4, y_ref);    // 参考答案
    fir_4_4_seq(x,N,h,y);       // 快速算法
    printf("[INF] error:%.4e\n",calc_err(y,y_ref+3,N-4));
    
    printf("[INF] ==== testing 4-tap (output 2) FIR fir_2_4()...\n");
    conv(x, N, h, 4, y_ref);    // 参考答案
    fir_2_4_seq(x,N,h,y);       // 快速算法
    printf("[INF] error:%.4e\n",calc_err(y,y_ref+3,N-4));

    printf("[INF] ==== testing 5-tap (output 2) FIR fir_2_5()...\n");
    conv(x, N, h, 5, y_ref);    // 参考答案
    fir_2_5_seq(x,N,h,y);       // 快速算法
    printf("[INF] error:%.4e\n",calc_err(y,y_ref+4,N-4));
    
    return 0;
}
