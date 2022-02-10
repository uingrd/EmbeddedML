#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef CLANG_MACOS
    #include <malloc.h>
#endif

extern const int NUM_TAPS;          // 滤波器抽头数目
extern const int Y_SHIFT;           // 输出数据格式转换移位量
extern const int NUM_X;             // 测试数据长度
extern const signed char x_int[];   // 存放测试数据
extern const signed char taps_int[];// 存放滤波器抽头

int8_t *buf=0;  // 环形缓冲器，存放滑动窗口内的待滤波数据
int p=0;        // 环形缓冲器指针
int8_t fir(int8_t x)
{
    // 数据进入环形缓冲器
    buf[p]=x;   
    p=(p+1)%NUM_TAPS;
    // 计算FIR滤波输出
    int32_t sum=0;  
    for (int i=0; i<NUM_TAPS; i++)
        sum+=((int32_t)buf[(p+i)%NUM_TAPS])*((int32_t)taps_int[i]);
    // 输出格式调整
    sum>>=Y_SHIFT;          
    // 饱和运算
    if (sum>127) sum=127;   
    if (sum<-128) sum=-128;
    return (int8_t)sum;
}

int main()
{
    FILE* fp;
    int8_t y;
# ifdef CLANG_MACOS
    fp = fopen("y_int.bin", "wb");
#else
    fopen_s(&fp, "y_int.bin", "wb");
#endif
   
    buf=(int8_t*)calloc(NUM_TAPS,1);
 
    for (int n=0; n<NUM_X; n++)
    {
        y=fir(x_int[n]);
        fwrite(&y,1,1,fp);
    }
    fclose(fp);
    free(buf);

    return 0;
}

