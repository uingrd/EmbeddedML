#include <stdio.h>
#include <math.h>

#define MAX2(a,b) (((a)>(b))?(a):(b))

// 几种近似tanh计算方法
// 代码清单 2 - 12
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

float tanh_3(float x)
{
    if (x>(float) 3.44540506f) return  1;
    if (x<(float)-3.44540506f) return -1;
    
    x*= 1.0f/3.44540506f;
    x*= fabsf(x)-2.00002276f;
    return x*(fabsf(x)-2.0000286f);
}

int main()
{
    float e1=0, e2=0, e3=0, y0, y1, y2, y3;

    for (float x=-3.5f; x<3.5f; x+=1e-3f)
    {
        y0=(float)tanh(x);
        y1=tanh_1(x);
        y2=tanh_2(x);
        y3=tanh_3(x);
        
        e1=MAX2(e1,fabsf(y0-y1));
        e2=MAX2(e2,fabsf(y0-y2));
        e3=MAX2(e3,fabsf(y0-y3));
    }
    printf("[INF] tanh_1, difference: %e\n",e1);
    printf("[INF] tanh_2, difference: %e\n",e2);
    printf("[INF] tanh_3, difference: %e\n",e3);
    return 0;
}
