# AlphaTensor矩阵快速乘法的计算表达式提取和打印

下面是运行该程序时打印输出内容的例子

&emsp;&emsp;size:(2,2,3), rank:11, (12, 91.66666666666666%)<br>
&emsp;&emsp;-----------------------------------------------<br>
&emsp;&emsp;p[0] = x[1,0] * (y[0,1] - y[1,1])<br>
&emsp;&emsp;p[1] = (x[1,0] + x[1,1]) * y[1,1]<br>
&emsp;&emsp;p[2] = (x[0,1] + x[1,0]) * (y[0,2] + y[1,1])<br>
&emsp;&emsp;p[3] = x[0,1] * (y[0,2] - y[1,2])<br>
&emsp;&emsp;p[4] = (x[0,0] + x[0,1]) * y[0,2]<br>
&emsp;&emsp;p[5] = (x[0,0] - x[1,0]) * (y[0,1] + y[0,2])<br>
&emsp;&emsp;p[6] = x[0,0] * (y[0,0] - y[0,2])<br>
&emsp;&emsp;p[7] = (x[0,1] - x[1,1]) * (y[1,1] + y[1,2])<br>
&emsp;&emsp;p[8] = (x[1,0] - x[1,1]) * y[0,0]<br>
&emsp;&emsp;p[9] = x[0,1] * (y[0,2] - y[1,0])<br>
&emsp;&emsp;p[10] = x[1,1] * (y[0,0] + y[1,0])<br>
&emsp;&emsp;z[0,0] = p[4] + p[6] - p[9]<br>
&emsp;&emsp;z[0,1] = p[0] + p[2] - p[4] + p[5]<br>
&emsp;&emsp;z[0,2] = -p[3] + p[4]<br>
&emsp;&emsp;z[1,0] = p[10] + p[8]<br>
&emsp;&emsp;z[1,1] = p[0] + p[1]<br>
&emsp;&emsp;z[1,2] = -p[1] + p[2] - p[3] - p[7]<br>

上述内容表明打印的是2x2矩阵x和2x3矩阵y的快速乘法算法，计算结果存储于2x3矩阵z。计算过程中中间数据存储于数组p

程序运行需要的文件factorizations_r.npz来自：https://github.com/deepmind/alphatensor/tree/main/algorithms

程序配置参数为：<br>
&emsp;&emsp;验证的矩阵乘法分解式数目，如果想验证所有分解式的话，可以设置为np.inf<br>
&emsp;&emsp;NUM_EQU_VERIFY=10<br>
&emsp;&emsp;是否打印矩阵计算表达式<br>
&emsp;&emsp;PRINT_EQU=True
