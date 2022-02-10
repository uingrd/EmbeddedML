import platform,os,sys
# 设置当前工作目录
os.chdir(sys.path[0])

if platform.system()=='Darwin': # MacOS
    os.system('clang mat_f32.c -Wall -c -o export_code/mat_f32.o -fno-stack-protector')
    os.system('clang export_code/percep_test.c -Wall -c -o export_code/percep_test.o -fno-stack-protector')
    os.system('clang export_code/mat_f32.o export_code/percep_test.o -Wall -o export_code/test')
    os.system('export_code/test')
else:
    os.system('gcc mat_f32.c -Wall -c -o export_code/mat_f32.o -fno-stack-protector')
    os.system('gcc export_code/percep_test.c -Wall -c -o export_code/percep_test.o -fno-stack-protector')
    os.system('gcc export_code/mat_f32.o export_code/percep_test.o -Wall -o export_code/test')
    os.system('export_code/test')
