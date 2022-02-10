import platform,os,sys
# 设置当前运行目录
os.chdir(sys.path[0])

if platform.system()=='Darwin': # MacOS
    os.system('clang fast_fir.c -Wall -c -o export_code/fast_fir.o -fno-stack-protector')
    os.system('clang export_code/fast_fir.o -Wall -o export_code/test_fir')
    os.system('export_code/test_fir')

    os.system('clang fast_conv_1d.c -Wall -c -o export_code/fast_conv_1d.o -fno-stack-protector')
    os.system('clang export_code/fast_conv_1d.o -Wall -o export_code/test_conv_1d')
    os.system('export_code/test_conv_1d')
else:
    os.system('gcc fast_fir.c -Wall -c -o export_code/fast_fir.o -fno-stack-protector')
    os.system('gcc export_code/fast_fir.o -Wall -o export_code/test_fir')
    os.system('export_code/test_fir')

    os.system('gcc fast_conv_1d.c -Wall -c -o export_code/fast_conv_1d.o -fno-stack-protector')
    os.system('gcc export_code/fast_conv_1d.o -Wall -o export_code/test_conv_1d')
    os.system('export_code/test_conv_1d')
