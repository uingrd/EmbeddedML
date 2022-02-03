
import sys,os

os.system('clang mat_f32.c -Wall -c -o export_code/mat_f32.o -fno-stack-protector')
os.system('clang svm_rbf.c -Wall -c -o export_code/svm_rbf.o -fno-stack-protector')
os.system('clang export_code/svm_rbf_test.c -Wall -c -o export_code/svm_rbf_test.o -fno-stack-protector')
os.system('clang export_code/mat_f32.o export_code/svm_rbf.o export_code/svm_rbf_test.o -Wall -o export_code/test')
os.system('export_code/test')
