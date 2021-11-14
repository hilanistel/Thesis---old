import pandas as pd
from docplex.cp.model import CpoModel
import datetime
import time

# open file to save rum results
start = time.time()
filename1 = r'NewFormulation Cancer Diff B '+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
outFileName=r"C:\Users\hila\Desktop\runs\ " + filename1 + r".txt"
outFile=open(outFileName, "w")

# read data from computer
data = pd.read_csv(r"C:\Users\hila\Desktop\data\cancer.csv", header=None)
N = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
data = data.drop(0, axis =1)
X =data.drop(1, axis=1) # drop the reading class
label =data[1]
data_readings = [tuple(row) for row in X.values]
TR = {}

# create dict of readings and labels
for index_i, i in enumerate(data_readings):
    TR.update({i:{}})
    for index_j, j in enumerate(label):
      if index_j == index_i:
        if label[index_j] == 'M':
          TR[i] = 1
        if label[index_j] == 'B':
          TR[i] = 0

#std andd mean of all colmuns
threshold = {1:0}
for i in range (2,32):
  threshold.update({i:{}})
  threshold[i] = (data[i].describe()['std'])/10
  data[i].fillna(data[i].describe()['mean'])

# print threshold to file
outFile.write("""threshold\n""")
outFile.write(str(threshold) + '\n')
print(threshold)

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

# print one pair of readings to file
outFile.write("""one pair of readings\n""")
outFile.write(str(p[0]) + '\n')
print(p[0])

# dict of arrays with size = number of features
f = {}
for i in N:
  f.update({i:[]})
print(f)
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
      if (i - rq[1][index] > threshold[index+1]) :
        if rq not in f[index+1]:
          f[index+1].append(rq)
          objective[index][j] = 1
  return(f)
F(p)
print("end f and objective")

def Yrq(rq):
  yrq = 0
  for i in f:
    if rq in f[i]:
      yrq += 1
  return yrq
# print diff in features between rq
outFile.write("""diff in features between rq\n""")
x = Yrq([(17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05372999999999999, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189), (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06663999999999999, 0.047810000000000005, 0.1885, 0.05766, 0.2699, 0.7886, 2.0580000000000003, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.14400000000000002, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)])
outFile.write(str(x) + '\n')

# check if featur i diff in readings rq
def Iirq(i, rq):
  if rq in f[i]:
    return 1
  return 0
outFile.write("""is featur i diff in readings rq\n""")
y = Iirq(3,[(17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05372999999999999, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189), (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06663999999999999, 0.047810000000000005, 0.1885, 0.05766, 0.2699, 0.7886, 2.0580000000000003, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.14400000000000002, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)])
outFile.write(str(y))


y ={}
for index_j,j in enumerate(objective):
  for index,i in enumerate(j):
    if index not in y.keys():
      y.update({index:[]})
    if i > 0:
      for x in N:
        if index_j == x:
          y[index].append('x%d' %x)
print(y[0])


# define B = number of features we want to select
f_num = [5,11]
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
            y[index].append(globals()["x" + str(index_j+1)])
    obj = []
    for i in range(len(p)):
        obj.append(mdl.log(mdl.sum(y[i]))+1)


    mdl.maximize(mdl.sum(obj))

    msol=mdl.solve()
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