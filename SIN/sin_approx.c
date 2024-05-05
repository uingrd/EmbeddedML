////////////////////////
// 近似sin计算方法
////////////////////////

#define PI 3.1415926535897932384626433832795
#define SIGN(x) (((x)>=0)?((x)>0):-1)
#define ABS(x) ((x)>0?(x):(-(x)))

float algo1(float x)
{
    float y=x/(float)PI;
    return 4.0f*y*(1.0f-y);
}

float algo2(float x)
{
//    return -0.417698f*x*x+1.312236f*x-0.050465f;
    return (float)(60.0*(PI*PI-12.0)/(PI*PI*PI*PI*PI))*x*x-
           (float)(60.0*(PI*PI-12.0)/(PI*PI*PI*PI))*x+
           (float)(12.0*(PI*PI-10.0)/(PI*PI*PI));
}

float algo3(float x)
{
    return 16.0f*x*((float)PI-x)/((float)(5.0*PI*PI)-4.0f*x*((float)PI-x));
}
 
typedef float (*algo_t)(float);

// 计算ARCTAN近似，使用不同的算法
// 输入数据范围是-PI~+PI
float sin_approx(float x, algo_t algo)
{
    if ((x>(float)(PI*2.0f)) || (x<-(float)(PI*2.0f)))
    {
        int k=x/(float)(PI*2.0f);
        x-=(float)k*(float)(PI*2.0);
    }
    if (x>PI) x-=(float)(PI*2.0);
    
    return SIGN(x)*algo(ABS(x));
}

////////////////////////
// 单元测试
////////////////////////

#include <math.h>
#include <stdio.h>
#define N 10000
#define MAX(a,b) (((a)>(b))?(a):(b))

// 测试近似算法误差
float test_algo(algo_t algo)
{
    float t,err=0;
    for (int n=0; n<N; n++)
    {
        t=(float)(n-N/2)/(float)(N+1)*(float)M_PI*2.0f;
        err=MAX(err,ABS(sin_approx(t,algo)-sin(t)));
    }
    return err;
}

int main()
{
    printf("algo1 err. max: %f\n", test_algo(algo1));
    printf("algo2 err. max: %f\n", test_algo(algo2));
    printf("algo3 err. max: %f\n", test_algo(algo3));
    
    return 0;
}
