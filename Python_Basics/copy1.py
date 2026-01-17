import copy
a = [10,1,2,[2,3,4]]
b = a 
b.append(11)
print(b)
b[3][1] = 99 
print(a)
print(b)
b = copyOf_a = a.copy() 
b.append(12)
print(a)
print(b)
b = copy.deepcopy(a)
b[3][0] = 88
print(a)
print(b)