import random

file_handle1=open('Input-P1.list',mode='w')
for i in range(20):
    file_handle1.write(str(float(random.randint(1,10000)+round(random.random(),18))) + '\n')

file_handle1=open('Input-P2.list',mode='w')
for i in range(20):
    file_handle1.write(str(float(random.randint(5000,9000)+round(random.random(),18))) + '\n')