import random

file_handle1=open('Input-P0.list',mode='w')
for i in range(100):
    file_handle1.write(str(round(random.random(),6)) + '\n')

file_handle1=open('Input-P1.list',mode='w')
for i in range(100):
    file_handle1.write(str(round(random.random(),6)) + '\n')

file_handle1=open('Input-P2.list',mode='w')
for i in range(100):
    file_handle1.write(str(round(random.random(),6)) + '\n')