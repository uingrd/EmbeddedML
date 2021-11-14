#include <stdio.h>
#include "./export_code/param.h"
#include "./export_code/test_data.h"

// gcc -o mnist mnist.c export_code/param.c export_code/test_data.c -std=c99

// 2D卷积层计算
void conv2d_f32(float* dout,        // 输出数据
                float* din,         // 输入数据
                int din_hgt,        // 输入数据(矩阵)高度
                int din_wid,        // 输入数据(矩阵)宽度
                const float* ker,   // 卷积核
                const float* bias,  // 偏置
                const int *shape)   // 卷积核形状
{
    int din_size = din_hgt * din_wid;

    // 卷积核尺寸
    int num_cout = shape[0], num_cin = shape[1], k_hgt = shape[2], k_wid = shape[3];
    int k_size = k_hgt * k_wid; // 单个卷积和矩阵尺寸

    // 输出数据尺寸
    // dout[num_cout][dout_hgt][dout_wid]存放输出特征图数据
    int dout_hgt = (din_hgt - k_hgt + 1);
    int dout_wid = (din_wid - k_wid + 1);
    int dout_size = dout_hgt * dout_wid; // 单个输出特征图矩阵尺寸

    const float* din_sel;
    const float* ker_sel;
    const float* ker_elm;
    const float* din_elm;
    const float* din_elm0; 
    float* dout_sel;
    float* dout_elm;

    ker_sel = ker;  // 该指针跟踪每次卷积使用的卷积核矩阵
    dout_sel=dout;  // 该指针跟踪cout通道输出数据
    for (int cout = 0; cout < num_cout; cout++, 
                                        dout_sel += dout_size)  // 指向cout通道输出数据
    {
        // 加上偏置
        for (int n = 0; n < dout_size; n++)
            dout_sel[n] = bias[cout];
        
        din_sel = din;
        for (int cin = 0; cin < num_cin; cin++, 
                                         din_sel += din_size,   // 指向cin通道输入数据
                                         ker_sel += k_size)     // 指向每次卷积使用的卷积核矩阵
        {
            ker_elm = ker_sel;                                  // 跟踪当前卷积对应的卷积核元素
            din_elm0 = din_sel;
            for (int kh = 0; kh < k_hgt; kh++,                  // kh对应卷积核内部元素行号        
                                         din_elm0+=din_wid)     // 指向数据滑动窗内和ker_elm对应的数据行的第一个数据位置         
            {
                for (int kw = 0; kw < k_wid; kw++,              // kw对应卷积核内部元素列号
                                             ker_elm++)         // ker_elm指向元素内(kh, kw)位置元素
                {                                               
                    if (!*ker_elm) continue;

                    din_elm = &din_elm0[kw];                    // 指向数据滑动窗内和ker_elm对应的数据位置
                    dout_elm = dout_sel;
                    for (int h = 0; h < dout_hgt; h++,
                                                  din_elm += din_wid)   // 指向数据滑动窗的下一行数据和ker_elm对应的数据位置
                    {
                        for (int w = 0; w < dout_wid; w++,
                                                      dout_elm++)
                        {
                            *dout_elm+= din_elm[w] * (*ker_elm);
                        }
                    }
                }
            }

        }
    }
    return;
}


// 2D卷积层计算(数据访问用到了乘法的“慢速版本”)
void conv2d_f32_slow(float* dout,        // 输出数据
    float* din,         // 输入数据
    int din_hgt,        // 输入数据(矩阵)高度
    int din_wid,        // 输入数据(矩阵)宽度
    const float* ker,   // 卷积核
    const float* bias,  // 偏置
    const int* shape)   // 卷积核形状
{
    // 卷积核尺寸
    int num_cout = shape[0], num_cin = shape[1], k_hgt = shape[2], k_wid = shape[3];
 
    // 输出数据尺寸
    int dout_hgt = (din_hgt - k_hgt + 1);
    int dout_wid = (din_wid - k_wid + 1);

    for (int cout = 0; cout < num_cout; cout++)
    {
        // 加上偏置
        for (int n = 0; n < dout_hgt * dout_wid; n++)
            dout[cout * dout_hgt * dout_wid + n] = bias[cout];
        // 对每个输入通道计算2D卷积
        for (int cin = 0; cin < num_cin; cin++)
        {
            // h和w是滑动窗位置
            for (int h = 0; h < dout_hgt; h++)
            {
                for (int w = 0; w < dout_wid; w++)
                {
                    // kh和kw是卷积核内的元素位置
                    for (int kh = 0; kh < k_hgt; kh++)
                    {
                        for (int kw = 0; kw < k_wid; kw++)
                        {
                            dout[cout * dout_hgt * dout_wid + h * dout_wid + w] +=                          // dout[cout][h][w]
                                din[cin * din_hgt * din_wid + (h + kh) * din_wid + (w + kw)] *              // din[cin][h+kh][w+kw]
                                ker[cout * num_cin * k_hgt * k_wid + cin * k_hgt * k_wid + kh * k_wid + kw];// ker[cout][cin][kh][kw]
                        }
                    }
                }
            }
        }
    }
    return;
}


// 全连接层运算
void fc_f32(float* dout,        // 输出数据
            float* din,         // 输入数据
            const float* weight,// 权重
            const float* bias,  // 偏置
            const int* shape)   // 权重矩阵形状
{
    // 数据尺寸
    int num_cout = shape[0], num_cin = shape[1];
    const float* w = weight;
    const float* d;
    for (int cout = 0; cout < num_cout; cout++)
    {
        dout[cout] = bias[cout];
        d = din;
        for (int cin = 0; cin < num_cin; cin++, w++, d++)
            if (*w) dout[cout] += (*w) * (*d);
    }
}


// 全连接层运算(数据访问用到了乘法的“慢速版本”)
void fc_f32_slow(float* dout,        // 输出数据
    float* din,         // 输入数据
    const float* weight,// 权重
    const float* bias,  // 偏置
    const int* shape)   // 权重矩阵形状
{
    // 数据尺寸
    int num_cout = shape[0], num_cin = shape[1];
    for (int cout = 0; cout < num_cout; cout++)
    {
        dout[cout] = bias[cout];
        for (int cin = 0; cin < num_cin; cin++)
            dout[cout] += weight[cout * num_cin + cin] * din[cin];
    }
}

void maxpool2d(float* dout, // 输出数据
               float* din,  // 输入数据
               int din_hgt, // 输入数据(矩阵)高度
               int din_wid, // 输入数据(矩阵)宽度
               int num_c,   // 通道数
               int ksize)   // 窗口尺寸
{
    int dout_hgt = 1 + (din_hgt - ksize) / ksize;
    int dout_wid = 1 + (din_wid - ksize) / ksize;
    float m,v;
    float* din_sel;

    for (int c = 0; c < num_c; c++)
    {
        for (int h = 0; h < dout_hgt; h++)
        {
            for (int w = 0; w < dout_wid; w++)
            {
                din_sel = &din[c * din_hgt * din_wid + h * ksize * din_wid + w * ksize];
                m = din_sel[0];
                for (int y = 0; y < ksize; y++)
                {
                    for (int x = 0; x < ksize; x++)
                    {
                        v = din_sel[y * din_wid + x];
                        if (v > m) m = v;
                    }
                }
                dout[c * dout_hgt * dout_wid + h * dout_wid + w] = m;
            }
        }
    }
    return;
}


void relu(float* dout, float* din, int size)
{
    for (int n = 0; n < size; n++)
    {
        dout[n] = din[n] > 0 ? din[n] : 0;
    }
}


float calc_error(float* p1, float* p2, int size)
{
    float e = 0, v;
    for (int n = 0; n < size; n++)
    {
        v = p1[n] - p2[n];
        if (v < 0) v = -v;
        if (v > e) e=v;
    }
    return e;
}

float buf0[32 * 24 * 24];
float buf1[32 * 24 * 24];
float buf2[32 * 24 * 24];

int calc(float* din)
{
    int res;
    float vmax;

    // x = self.conv1(din)       # (1, 28, 28)->(32, 24, 24)
    conv2d_f32(buf0,                    // 输出数据
               din,                     // 输入数据
               28,28,                   // 输入数据(矩阵)高度/宽度
               conv1_weight,            // 卷积核
               conv1_bias,              // 偏置
               conv1_weight_shape);     // 卷积核形状
    // x = F.relu(x)
    relu(buf0, buf0, 32 * 24 * 24);
    // x = F.max_pool2d(x, 2)  # (32, 24, 24)->(32, 12, 12)
    maxpool2d(buf1, buf0, 24, 24, 32, 2);
    // x = self.conv2(x)       # (32, 12, 12)->(32, 8, 8)
    conv2d_f32(buf0,                // 输出数据
               buf1,                // 输入数据
               12, 12,              // 输入数据(矩阵)高度/宽度
               conv2_weight,        // 卷积核
               conv2_bias,          // 偏置
               conv2_weight_shape); // 卷积核形状
    // x = F.relu(x)
    relu(buf0, buf0, 32 * 8 * 8);
    // x = F.max_pool2d(x, 2)  # (N, 32, 8, 8)->(N, 32, 4, 4)
    maxpool2d(buf1, buf0, 8, 8, 32, 2);
    // x = torch.flatten(x, 1) # (N, 32, 4, 4)->(N, 512)
    // x = self.fc1(x)         # (N, 512)->(N, 1024)
    fc_f32(buf0, buf1, fc1_weight, fc1_bias, fc1_weight_shape);
    // x = F.relu(x)
    relu(buf0, buf0, 1024);
    // x = self.fc2(x)         # (N, 1024)->(N, 10)
    fc_f32(buf1, buf0, fc2_weight, fc2_bias, fc2_weight_shape);
    
    // argmax
    res = 0;
    vmax = buf1[0];
    for (int n = 1; n < 10; n++)
        if (buf1[n] > vmax)
        {
            vmax = buf1[n];
            res = n;
        }
    return res;
}

int main()
{
    const float* din;
    int res,err;
    err = 0;
    for (int n = 0; n < TEST_DATA_NUM; n++)
    {
        din = &test_x[n][0];
        res = calc((float*)din);
        err += res != test_y[n];
        printf("[INF] test: %d, output: %d, reference: %d %s\n", n, res, test_y[n],(res==test_y[n])?"":"******");
    }
    printf("[INF] #error: %d, ACC: %.2f%%\n", err, 100.0-(float)err / (float)TEST_DATA_NUM * 100.0);
	return 0;
}
