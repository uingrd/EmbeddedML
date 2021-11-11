#include <stdio.h>
#include <math.h>

// 几种近似tanh计算方法
// 代码清单 2 - 12

float tanh_0(float x)
{
    float y=expf(x+x);
    return (y-1.0f)/(y+1.0f);
}

float tanh_1(float x)
{
    if (x < -3) return -1;
    if (x >  3) return  1;
    return x*(27.0f+x*x)/(27.0f+9.0f*x*x);
}

float tanh_2(float x)
{
    if (x>(float) 3.4f) return  1;
    if (x<(float)-3.4f) return -1;
    
    x*= 1.0f/3.4f;
    x*= fabsf(x)-2.0f;
    return x*(fabsf(x)-2.0f);
}

int main()
{
    float err=0, y0, y1;
    for (float x=-3.5f; x<3.5f; x+=1e-3f)
    {
        y0=(float)tanh(x);
        y1=tanh_2(x);
        float e=fabsf(y0-y1);
        if (e>err)
            err=e;
    }
    printf("[INF] error: %e\n",err);
    return 0;
}
