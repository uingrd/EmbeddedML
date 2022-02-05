#include "float_shift.h"

// 通过指数字段操作，计算浮点数x*pow(2,s)
float float_shift(float x, int s)
{
    unsigned long* y = (unsigned long*)&x;
    *y += s << 23; // 对应单精度浮点数指数域：23~30-bit 
    return *(float*)y;
}

// 通过指数字段操作，计算双精度浮点数x*pow(2,s)
double double_shift(double x, int s)
{
    unsigned long *y= (unsigned long*)&x;
    y[1] += s << 20; // 对应双精度浮点数指数域：52~62-bit
    return *((double*)y);
}

