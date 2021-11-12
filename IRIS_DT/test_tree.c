
#include "tree.h"
#include "test_tree_dat.h"

int tree(float *feature, int *target);

const float dt_tree_test_in [NUM_DAT*NUM_DIM];
const int   dt_tree_test_out[NUM_DAT];

float test_tree()
{
    int score_max;
    int idx_max;
    int err=0;
    
    int score[NUM_CLS];
    for (int n=0; n<NUM_DAT; n++)
    {   // &dt_tree_test_in[n*NUM_DIM]指向NUM_DIM维的测试输入向量
        // score数组存放每个类型的得分
        tree((float *)&dt_tree_test_in[n*NUM_DIM], (int*)score);
        
        // 找出得分最高的类别
        score_max=score[0];
        idx_max=0;
        for (int m=1; m<NUM_CLS; m++)
            if (score[m]>score_max)
            {
                score_max=score[m];
                idx_max=m;
            }
        // 比较参考答案，并对错误计数
        if (dt_tree_test_out[n]!=idx_max) err++;
    }
    // 返回错误率
    return (float)err/(float)NUM_DAT;
}

#include <stdio.h>
int main(void)
{
    float res=test_tree();
    printf("[INF] acc: %.4f%%\n",(1.0-res)*100.0);
}

