import platform,os,sys
# 设置当前运行目录
os.chdir(sys.path[0])

if platform.system()=='Darwin': # MacOS
    os.system('clang gnb.c -Wall -c -o export_code/gnb.o -fno-stack-protector')
    os.system('clang export_code/gnb_test.c -Wall -c -o export_code/gnb_test.o -fno-stack-protector')
    os.system('clang export_code/gnb.o export_code/gnb_test.o -Wall -o export_code/test')
    os.system('export_code/test')
else:
    os.system('gcc gnb.c -Wall -c -o export_code/gnb.o -fno-stack-protector')
    os.system('gcc export_code/gnb_test.c -Wall -c -o export_code/gnb_test.o -fno-stack-protector')
    os.system('gcc export_code/gnb.o export_code/gnb_test.o -Wall -lm -o export_code/test')
    os.system('export_code/test')
