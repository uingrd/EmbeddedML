
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


// (2,2)线性卷积 
//   计算{x[0],x[1]}和{h[0],h[1]}的线性卷积
//   输出{y[0],y[1],y[2]}
void conv_2_2(float *x, float *h, float *y)
{
    float ha=h[1]-h[0];
        
    float ya=(x[0]-x[1])*ha;  
    y[0]=x[0]*h[0];
    y[2]=x[1]*h[1];
    y[1]=y[0]+y[2]+ya;
}


// (3,2)线性卷积 
//   计算{x[0],x[1],x[2]}和{h[0],h[1]}的线性卷积
//   输出{y[0],y[1],y[2],y[3]}
void conv_3_2(float *x, float *h, float *y)
{
    float ha=float_shift(h[0]+h[1],-1);
    float hb=float_shift(h[0]-h[1],-1);
        
    float xc=x[0]+x[2];
    float xa=xc+x[1];
    float xb=xc-x[1];
        
    float m1=ha*xa;
    float m2=hb*xb;
        
    y[0]=h[0]*x[0];
    y[3]=h[1]*x[2];
    
    y[1]=m1-m2-y[3];
    y[2]=m1+m2-y[0];
}


// (3,3)线性卷积 
//   计算{x[0],x[1],x[2]}和{h[0],h[1],h[2]}的线性卷积
//   输出{y[0],y[1],y[2],y[3],y[4]}
// flag=0表明需要从卷积核计算中间变量g
// 注意：当需要多次卷积时g的值可以预先保存下来，不需要重复计算，调用时设置flag=1会复用输入的g数组
void conv_3_3(float *x, float *h, float *y, float *g, int flag)
{
    if (!flag)
    {
        g[0]= h[0]/2.0;          
        g[1]=(h[0]+h[1]+h[2])/2.0;
        g[2]=(h[0]-h[1]+h[2])/6.0;
        g[3]=(h[0]+2*h[1]+4*h[2])/6.0;
    }   
    float d0=x[1]+x[2];
    float d1=x[2]-x[1];
    float d2=x[0]+d0;
    float d3=x[0]+d1;
    float d4=d0+d0+d1+d2;
        
    float m0=g[0]*x[0];
    float m1=g[1]*d2;
    float m2=g[2]*d3;
    float m3=g[3]*d4;
        
    y[4]=h[2]*x[2];
        
    float u0=m1+m1;
    float u1=m2+m2;
    float u2=y[4]+y[4]-m0-m3;
    float u3=m1+m2;
        
    y[0]= m0+m0;
    y[1]= u0-u1+u2;
    y[2]= u1+u3-y[0]-y[4];
    y[3]=-u2-u3;
}

// (3,3)线性卷积（增加1次乘法换取加法次数的减少）
//   计算{x[0],x[1],x[2]}和{h[0],h[1],h[2]}的线性卷积
//   输出{y0,y1,y2,y3,y4}
// flag=0表明需要从卷积核计算中间变量g
// 注意：当需要多次卷积时g的值可以预先保存下来，不需要重复计算，调用时设置flag=1会复用输入的g数组
void conv_3_3a(float *x, float *h, float *y, float *g, int flag)
{
    // 当需要多次卷积时，g0-g2的值可以预先保存下来，不需要重复计算
    if (!flag)
    {
        g[0]=h[0]+h[1];
        g[1]=h[0]+h[2];
        g[2]=h[1]+h[2];
    }
    
    float d0=x[0]+x[1];
    float d1=x[0]+x[2];
    float d2=x[1]+x[2];
    
    float m0=h[1]*x[1];
    float m1=g[0]*d0;
    float m2=g[1]*d1;
    float m3=g[2]*d2;
    
    y[0]=h[0]*x[0];
    y[4]=h[2]*x[2];
    
    y[1]=m1-m0-y[0];
    y[2]=m2+m0-y[0]-y[4];
    y[3]=m3-m0-y[4];
}

// (3,3)线性卷积 
//   计算{x[0],x[1],x[2]}和{h[0],h[1],h[2]}的线性卷积
//   输出{y0,y1,y2,y3,y4}
// flag=0表明需要从卷积核计算中间变量g
// 注意：当需要多次卷积时g的值可以预先保存下来，不需要重复计算，调用时设置flag=1会复用输入的g数组
void conv_3_3b(float *x, float *h, float *y, float *g, int flag)
{
    // 当需要多次卷积时，g0-g4的值可以预先保存下来，不需要重复计算
    if (!flag)
    {   
        g[0]= h[0]/2.0;          
        g[1]=(h[0]+h[1]+h[2])/2.0;
        g[2]=(h[0]-h[1]+h[2])/6.0;
        g[3]=(h[0]+2*h[1]+4*h[2])/6.0;
        g[4]= h[2];
    }
   
    float d0=x[0];
    float da=x[0]+x[2];
    float d1=da+x[1];
    float d2=da-x[1];
    float d3=x[0]+2*x[1]+4*x[2];
    float d4=x[2];
    
    float s0=g[0]*d0;
    float s1=g[1]*d1;
    float s2=g[2]*d2;
    float s3=g[3]*d3;
    float s4=g[4]*d4;
    
    y[0]= s0*2                  ;
    y[1]=-s0  +s1*2-s2*2-s3+s4*2;
    y[2]=-s0*2+s1  +s2*3   -s4  ;
    y[3]= s0  -s1  -s2  +s3-s4*2;
    y[4]=                   s4  ;
}


// (4,4)线性卷积，通过嵌入(2,2)线性卷积得到
//   计算{x[0],x[1],x[2],x[3]}和{h[0],h[1],h[2],h[3]}的线性卷积
//   输出{y0,y1,y2,y3,y4,y5,y6}
void conv_4_4(float *x, float *h, float *y)
{
    float a01=x[0]+x[1],a10=x[0]+x[2],a12=x[1]+x[3],a21=x[2]+x[3];
    float a11=a10+a12;
        
    // 应用时，如果h固定不变，可以把b预先计算后保存
    float b01=h[0]+h[1],b10=h[0]+h[2],b12=h[1]+h[3],b21=h[2]+h[3]; 
    float b11=b10+b12;
        
    y[0]=x[0]*h[0];
    float m01=a01*b01,m02=x[1]*h[1];
    float m20=x[2]*h[2],m21=a21*b21;
    y[6]=x[3]*h[3];
    float m10=a10*b10,m11=a11*b11,m12=a12*b12;
        
    float u0 = m02-m20;
    float u1 = m11-m10-m12;
        
    y[1] = m01-m02-y[0]; 
    y[2] = m10-y[0]+u0;     
    y[5] = m21-m20-y[6]; 
    y[3] = u1-y[1]-y[5];
    y[4] = m12-y[6]-u0;
}

// (4,2)线性卷积，通过拼接(2,2)线性卷积得到 
//   计算{x[0],x[1],x[2],x[3]}和{h[0],h[1]}的线性卷积
//   输出{y0,y1,y2,y3,y4}
void conv_4_2(float *x, float *h, float *y)
{
    float y2a;
    conv_2_2(x,h,y);
    y2a=y[2];
    conv_2_2(x+2,h,y+2);
    y[2]+=y2a;
}

// 通过拼接(3,2)线性卷积得到(6,2)线性卷积 
// 不考虑系数预计算的话，运算量是8次乘法和14次加法
void conv_6_2(float *x, float *h, float *y)
{
    float y3a;
    conv_3_2(x,h,y);
    y3a=y[3];
    conv_3_2(x+3,h,y+3);
    y[3]+=y3a;
}


// (5,3)线性卷积，通过拼接(3,2)和(3,3)线性卷积得到 
//   计算{x[0],x[1],x[2],x[3],x[4]}和{h[0],h[1],h[2]}的线性卷积
//   输出{y0,y1,y2,y3,y4,y5,y6}
void conv_5_3(float *x, float *h, float *y)
{
    float Y1[5],Y2[5],g[4];
    conv_3_2(h,x,Y1);
    conv_3_3(x+2,h,Y2,g,0);
    y[0]=Y1[0];
    y[1]=Y1[1];
    y[2]=Y1[2]+Y2[0];
    y[3]=Y1[3]+Y2[1];
    y[4]=Y2[2];
    y[5]=Y2[3];
    y[6]=Y2[4];
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

int  main()
{
    float x[6], h[6], y[11], y_ref[11],g[6];
    
    // 测试数据生成
    for (int n=0;n<6;n++) x[n]=randu(-100.0,100.0);
    for (int n=0;n<6;n++) h[n]=randu(-100.0,100.0);
    
    // 卷积测试
    printf("[INF] 测试(2,2)线性卷积\n");
    conv(x, 2, h, 2, y_ref);    // 参考答案
    conv_2_2(x,h,y);          // 快速算法
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,3));
    
    printf("[INF] 测试(3,2)线性卷积\n");
    conv(x, 3, h, 2, y_ref);    // 参考答案
    conv_3_2(x,h,y);
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,4));
    
    printf("[INF] 测试(3,3)线性卷积\n");
    conv(x, 3, h, 3, y_ref);    // 参考答案
    conv_3_3(x,h,y,g,0);
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,5));
    
    printf("[INF] 测试(3,3)a线性卷积（增加1次乘法换取加法次数的减少）\n");
    conv(x, 3, h, 3, y_ref);    // 参考答案
    conv_3_3a(x,h,y,g,0);
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,5));
    
    printf("[INF] 测试(3,3)b线性卷积\n");
    conv(x, 3, h, 3, y_ref);    // 参考答案
    conv_3_3b(x,h,y,g,0);
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,5));
    
    printf("[INF] 测试(4,4)线性卷积，通过嵌入(2,2)线性卷积得到\n");
    conv(x, 4, h, 4, y_ref);    // 参考答案
    conv_4_4(x,h,y);
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,7));
    
    printf("[INF] 测试(4,2)线性卷积，通过拼接(2,2)线性卷积得到\n");
    conv(x, 4, h, 2, y_ref);    // 参考答案
    conv_4_2(x,h,y);
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,5));
    
    printf("[INF] 测试(6,2)线性卷积，通过拼接(3,2)线性卷积得到\n");
    conv(x, 6, h, 2, y_ref);    // 参考答案
    conv_6_2(x,h,y);
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,7));
    
    
    printf("[INF] 测试(5,3)线性卷积，通过拼接(3,2)和(3,3)线性卷积得到\n"); 
    conv(x, 5, h, 3, y_ref);    // 参考答案
    conv_5_3(x,h,y);
    printf("[INF] error:%.4e\n",calc_err(y,y_ref,7));

    return 0;
}


    
    
