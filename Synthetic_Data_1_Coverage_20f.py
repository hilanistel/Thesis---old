# Synthetic Data 1: 100 features,
# weighted features: 80 with w = 0, 5 with w = 1,  5 with w = 0.75, 5 with w = 0.5, 5 with w = 0.25
# random values in range [-1,1]
# class 1 if multiplication > 0 and else 2
# 500 rows

from docplex.cp.model import CpoModel
import datetime
import time
import sys
start = time.time()
# define B = number of features we want to select
B = 10
# open file to save rum results
stdoutOrigin=sys.stdout
sys.stdout = open(r'C:\Users\hila\Desktop\runs\Synthetic_Data_1_Coverage_20_log.txt', "w")

filename1 = r'Synthetic_Data_1_Coverage_20 '+ str(B) + r' ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
outFileName=r"C:\Users\hila\Desktop\runs\ " + filename1 + r".txt"
outFile=open(outFileName, "w")

#Create Synthetic Dataset
import pandas as pd
import random
import numpy as np
# assign data of lists.
data = []
label = []
N = list(range(0, 20)) #N = [0,..,99]
# weighted features
W = {1:[2,13],0.75:[5,18]}
vec_w = np.zeros(20)
for i in W:
    for j in W[i]:
        vec_w[j] = i

"""
for i in range (0,500):#create rows
    row = [random.uniform(-1, 1) for i in range(20)]# 20 features between [-1,1]
    r = np.asarray(row)
    mul = np.sum(r * vec_w) #weighted multiplication
    label.append(1 if mul > 0 else 2) # class 1 if mul > 0 , else 2
    data.append(row)
# Create DataFrame.
X = pd.DataFrame(data)
# Print the output.
print(X)
TR = {}
"""

# X.to_csv(r"C:\Users\hila\Desktop\runs\Synthetic_Data_1_20f.csv", index = False)
X = pd.read_csv(r"C:\Users\hila\Desktop\runs\Synthetic_Data_1_20f.csv", header=None)
for row in X.values:#create rows
    r = np.asarray(row)
    mul = np.sum(r * vec_w) #weighted multiplication
    label.append(1 if mul > 1 else 2) # class 1 if mul > 0 , else 2
TR = {}

# create dict of readings and labels
i = 0
labelclass = {1:[],2:[]}
for row in X.values:
    r = tuple(row)
    TR.update({r:{}})
    TR[r] = label[i]
    if label[i] == 1:
      labelclass[1].append(r)
    else:
      labelclass[2].append(r)
    i = i + 1

#std andd mean of all colmuns
threshold = {}
for i in range (0,20):
  threshold.update({i:{}})
  threshold[i] = (X[i].describe()['std'])


# pairs of different readings
def P(TR):
  P = []
  for index_i,i in enumerate(TR):
    for index_j,j in enumerate(TR):
      if index_j > index_i:
        if TR[i] != TR[j]:
          P.append([i,j])
  return(P)
p = P(TR)
print("end pairs")

# print one pair of readings to file
outFile.write("""one pair of readings\n""")
outFile.write(str(p[0]) + '\n')


# dict of arrays with size = number of features
f = {}
for i in N:
  f.update({i:[]})
outFile.write("""features dict\n""")
outFile.write(str(f) + '\n')

# create objective function
objective = []
for i in N:
  objective.append([0] * len(p))


# calc how many different features between pair of reading with diff class
def F(p):
  for j,rq in enumerate(p):
    for index,i in enumerate(rq[0]):
      if ((i - rq[1][index] > threshold[index])):
        if rq not in f[index]:
          f[index].append(rq)
          objective[index][j] = 1
  return(f)
F(p)
print("end f and objective")

# define B = number of features we want to select
f_num = [4]
for B in f_num:
    mdl = CpoModel(name='GTCP_NEW')
    for i in N:
      globals()["x" + str(i)] = mdl.binary_var(name="x" + str(i))

    sum = 0
    for i in N:
        sum = sum  + globals()["x" + str(i)]
    mdl.add(sum <= B)


    y ={}
    for index_j,j in enumerate(objective):
      for index,i in enumerate(j):
        if index not in y.keys():
          y.update({index:[1]})
        if i > 0:
            y[index].append(globals()["x" + str(index_j)])
    obj = []
    for i in range(len(p)):
        obj.append(mdl.log(mdl.sum(y[i]))+1)
    print(y[0])


    mdl.maximize(mdl.sum(obj))

    msol=mdl.solve(TimeLimit=60000)
    outFile.write("""solver solution is \n""")
    print(msol)
    outFile.write(str(msol))
    outFile.write("""solution is \n""")
    for i in N:
      y = "x" + str(i) +"=" ,msol["x" + str(i)]
      print(y)
      outFile.write(str(y)+ '\n')

# calc run time
end = time.time()

# total time taken
outFile.write(r"Runtime of the program is " + str(round(end - start)) + " sec")
print(f"Runtime of the program is " + str(round(end - start)) + " sec")



#close file
outFile.close()

sys.stdout.close()
sys.stdout=stdoutOrigin