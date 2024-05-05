////////////////////////
// 近似arctan2计算方法
////////////////////////

#define PI 3.1415926535897932384626433832795
#define SIGN(x) (((x)>=0)?((x)>0):-1)
#define ABS(x) ((x)>0?(x):(-(x)))

float algo1(float i, float q) { return i*q/(i*i+0.28125f*q*q); }
float algo2(float i, float q) { return i*q/(i*i+0.28086f*q*q); }
float algo3(float i, float q)
{
    float x=q/i;
    return (float)(PI/4.0)*x+0.285f*x*(1.0f-ABS(x));
}
float algo4(float i, float q)
{
    float x=q/i;
    return (float)(PI/4.0)*x+0.273f*x*(1.0f-ABS(x));
}
float algo5(float i, float q)
{
    float x=q/i;
    return (float)(PI/4.0)*x+x*(0.186982f-0.191942f*x*x);
}
float algo6(float i, float q)
{
    float x=q/i;
    return (float)(PI/4.0)*x-x*(ABS(x)-1.0f)*(0.2447f+0.0663f*ABS(x));
}

typedef float (*algo_t)(float, float);

// 计算ARCTAN近似，使用不同的算法
// 输出数据范围是-PI~+PI
float arctan_approx(float i, float q, algo_t algo)
{
    if (ABS(i)>ABS(q))
        return (i>0)? algo(i,q):SIGN(q)*3.141592654f+algo(i,q);
    else
        return (q>0)? (float)(PI/2.0)-algo(q,i):-(float)(PI/2.0)-algo(q,i);
}

////////////////////////
// 单元测试
////////////////////////

#include <stdio.h>
#include <math.h>
#define N 10000
#define MAX(a,b) (((a)>(b))?(a):(b))

// 测试近似算法误差
float test_algo(algo_t algo)
{
    float t,ta,err=0;
    for (int n=0; n<N; n++) 
    {
        t=(float)(n-N/2)/(float)(N+1)*(float)M_PI*2.0f;
        ta=arctan_approx(cos(t),sin(t),algo);
        err=MAX(err,fabs(t-ta));
    }
    return err;
}

#define RAD2DEG(x) ((x)*(float)(180.0/M_PI))

int main()
{
    printf("algo1 err. max: %f(deg)\n", RAD2DEG(test_algo(algo1)));
    printf("algo2 err. max: %f(deg)\n", RAD2DEG(test_algo(algo2)));
    printf("algo3 err. max: %f(deg)\n", RAD2DEG(test_algo(algo3)));
    printf("algo4 err. max: %f(deg)\n", RAD2DEG(test_algo(algo4)));
    printf("algo5 err. max: %f(deg)\n", RAD2DEG(test_algo(algo5)));
    printf("algo6 err. max: %f(deg)\n", RAD2DEG(test_algo(algo6)));
    
    return 0;
}
