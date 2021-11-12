
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
    {   // &dt_tree_test_in[n*NUM_DIM]ָ��NUM_DIMά�Ĳ�����������
        // score������ÿ�����͵ĵ÷�
        tree((float *)&dt_tree_test_in[n*NUM_DIM], (int*)score);
        
        // �ҳ��÷���ߵ����
        score_max=score[0];
        idx_max=0;
        for (int m=1; m<NUM_CLS; m++)
            if (score[m]>score_max)
            {
                score_max=score[m];
                idx_max=m;
            }
        // �Ƚϲο��𰸣����Դ������
        if (dt_tree_test_out[n]!=idx_max) err++;
    }
    // ���ش�����
    return (float)err/(float)NUM_DAT;
}

#include <stdio.h>
int main(void)
{
    float res=test_tree();
    printf("[INF] acc: %.4f%%\n",(1.0-res)*100.0);
}

