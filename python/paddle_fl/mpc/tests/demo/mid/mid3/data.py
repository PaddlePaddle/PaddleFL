import random

file_handle1=open('Input-P0.list',mode='w')
for i in range(100):
    file_handle1.write(str(float(random.randint(1,10000)+round(random.random(),18))) + '\n')

file_handle1=open('Input-P1.list',mode='w')
for i in range(98):
    file_handle1.write(str(float(random.randint(800,7000)+round(random.random(),18))) + '\n')

file_handle1=open('Input-P2.list',mode='w')
for i in range(99):
    file_handle1.write(str(float(random.randint(500,5000)+round(random.random(),18))) + '\n')