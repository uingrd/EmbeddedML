
import sys,os

os.system('clang fast_fir.c -Wall -c -o export_code/fast_fir.o -fno-stack-protector')
os.system('clang export_code/fast_fir.o -Wall -o export_code/test')
os.system('export_code/test')
