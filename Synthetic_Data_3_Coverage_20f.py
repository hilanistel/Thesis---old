# Synthetic Data 3: 100 features,
# 96 is irrelevant to class, 4 relevant
# random values in range [0,100]
# class is by biggest feature value (class 0,1,2,3)
# 500 rows


from docplex.cp.model import CpoModel
import datetime
import time
start = time.time()
# define B = number of features we want to select
B = 4
# open file to save rum results
import sys
stdoutOrigin=sys.stdout
sys.stdout = open(r'C:\Users\hila\Desktop\runs\Synthetic_Data_3_Coverage_40_log.txt', "w")

filename1 = r'Synthetic_Data_3_Coverage_40 '+ str(B) + r' ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
outFileName=r"C:\Users\hila\Desktop\runs\ " + filename1 + r".txt"
outFile=open(outFileName, "w")

#Create Synthetic Dataset
import pandas as pd
import random
import numpy as np
# assign data of lists.
data = []
label = []
N = list(range(0, 40)) #N = [0,..,99]
# Groups A and B
Class_features = {3:0,18:1,23:2,37:4}

for i in range (0,500):#create rows
    row = [random.uniform(0, 100)  for i in range(40)]# 20 features between [-1,1]
    r = np.asarray(row)
    max = -1
    for i in Class_features:
        if row[i]>max:
            max = row[i]
            maxC = Class_features[i]
    label.append(maxC) # class
    data.append(row)
# Create DataFrame.
X = pd.DataFrame(data)
X.to_csv(r'C:\Users\hila\Desktop\runs\Synthetic_Data_3_40f '+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+r'.csv', index = False)
# Print the output.
print(X)
TR = {}
data_readings = [tuple(row) for row in X.values]

# create dict of readings and labels
for index_i, i in enumerate(data_readings):
    TR.update({i:{}})
    for index_j, j in enumerate(label):
      if index_j == index_i:
          TR[i] = j

#std andd mean of all colmuns
threshold = {}
for i in range (0,40):
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
    context = Context()

    # Default log verbosity
    context.verbose = 9

    msol=mdl.solve(TimeLimit=50000)
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