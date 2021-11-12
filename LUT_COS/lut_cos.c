#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>

const float LUT[]=
{
     1.0000f, 0.9848f,  0.9396f, 0.8660f,
     0.7660f, 0.6427f,  0.5000f, 0.3420f,
     0.1736f, 0.0000f, -0.1736f,-0.3420f,
    -0.5000f,-0.6427f, -0.7660f,-0.8660f,
    -0.9396f,-0.9848f, -1.0000f,-0.9848f,
    -0.9396f,-0.8660f, -0.7660f,-0.6427f,
    -0.5000f,-0.3420f, -0.1736f,-0.0000f,
     0.1736f, 0.3420f,  0.5000f, 0.6427f,
     0.7660f, 0.8660f,  0.9396f, 0.9848f, 1.0f
};

float lut_cos_deg(unsigned x)
{
    x%=360;
    unsigned u=x/10,v=x%10;
    float a=LUT[u], b=LUT[u+1];
    return (float)(a+(b-a)*((float)v)/10.0f);
}

int main()
{
    float err=0, y0, y1;
    for (unsigned x=0; x<360; x++)
    {
        y0=(float)cos((float)x*M_PI/180.0f);
        y1=lut_cos_deg(x);
        float e=fabsf(y0-y1);
        if (e>err)
            err=e;
    }
    printf("[INF] error: %e\n",err);
}
