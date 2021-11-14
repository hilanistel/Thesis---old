import pandas as pd
import math
import datetime

import time
start = time.time()

filename1 = r'Information Mushrooms '+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
outFileName=r"C:\Users\hila\Desktop\runs\ " + filename1 + r".txt"
outFile=open(outFileName, "w")

# read data
data = pd.read_csv(r"C:\Users\hila\Desktop\data\mushrooms.csv", header=None)

# information : entropy calc
def pi(data):
  dict = {}
  for row in data:
    if tuple(row) not in dict.keys():
      dict.update({tuple(row):1})
    else:
      dict[tuple(row)] = dict[tuple(row)] + 1
  for i in dict.keys():
    dict[i] = dict[i]/len(data)
  return(dict)

def filtering(df, limits_dic):
  cond = None
  # Build the conjunction one clause at a time
  for key, val in limits_dic.items():
      if cond is None:
          cond = df[key] == val
      else:
          cond = cond & (df[key] == val)
  return(df.loc[cond])

def data_split(data,*n):
  d = data[list(n)]
  d = [row for row in d.values]
  dsplit = []
  uniqe = []
  l = 0
  for row in d:
    if (not any(list(map(lambda x : list(row) == list(x), uniqe)))):
      uniqe.append(row)
  for u in uniqe:
    limits_dic = {}
    for i in n:
      limits_dic.update({i :{}})
    for index,i in enumerate(u):
      for indexj,j in enumerate(n):
        if index==indexj:
          limits_dic[j]=i
    dsplit.append(filtering(data, limits_dic))
  return(dsplit)

def H (xi,*x):
  H = 0
  if(x == ()):
    data_sets = data_split(data,xi)
  else:
    data_sets = data_split(data,*x)
  for data_set in data_sets:
    p_data_set = len(data_set)/len(data)
    if(x == ()):
      H = H - p_data_set*math.log(p_data_set,2)
    else:
      data_i = data_set[[xi,*x]]
      data_i = [row for row in data_i.values]
      p_i = pi(data_i)
      Hj = 0
      for j in p_i:
        Hj = Hj - p_i[j]*math.log(p_i[j],2)
      H = H + p_data_set*Hj
  return(H)

# mutual information
#calc the mutual informaion betweem columns x1,..,xn and column y in data
def I(data,y,*x):
  if (len(x) == 1):
    return (H(y) - H(y,*x)) # I(x1;y) = H(y)+H(y|x1)
  else:
    Info = I(data,y,*x[:-1]) + H(x[-1],*x[:-1]) - H(x[-1],*(*x[:-1], y))
    return Info

#find the subset with the highest Mutual Information
N = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# function create to get all N subsets
def sub_lists(l):
    base = []
    lists = [base]
    for i in range(len(l)):
        orig = lists[:]
        new = l[i]
        for j in range(len(lists)):
            lists[j] = lists[j] + [new]
        lists = orig + lists

    return lists

subsets = sub_lists(N)
subsets.remove([])

max = 0
max_set = []
for s in subsets:
  info = I(data,21,*s)
  if info > max:
    max = info
    max_set = s
print(max_set)
print(max)

outFile.write("""maximum information features are \n""")
outFile.write(str(max_set))
outFile.write("""\n maximum information value is \n""")
outFile.write(str(max))

# calc run time
end = time.time()

# total time taken
outFile.write(r"Runtime of the program is " + str(round(end - start)) + " sec")
print(f"Runtime of the program is " + str(round(end - start)) + " sec")

#close file
outFile.close()