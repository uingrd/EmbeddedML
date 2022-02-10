import platform,os,sys
# 设置当前运行目录
os.chdir(sys.path[0])

FNAME_DLL='libAdd.so'
SRC=['main.cpp']
cmd='g++ -fPIC -shared -o '

cmd+=' '+FNAME_DLL
for s in SRC:
    cmd +=' '+s
print(cmd)

os.system(cmd)
