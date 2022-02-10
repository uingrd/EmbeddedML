import platform,os,sys
# 设置当前运行目录
os.chdir(sys.path[0])

if platform.system()=='Darwin': # MacOS
    os.system('clang export_code/auto_code.c -Wall -c -o export_code/auto_code.o -fno-stack-protector')
    os.system('clang export_code/auto_code.o -Wall -o export_code/test')
    os.system('export_code/test')
else:
    os.system('gcc export_code/auto_code.c -Wall -c -o export_code/auto_code.o -fno-stack-protector')
    os.system('gcc export_code/auto_code.o -Wall -o export_code/test')
    os.system('export_code/test')

