import pickle
f = open("./labels.txt", "r")
s = f.readline()
s = s.replace("[", "")
s = s.replace("]", "")
s = s.replace("'", "")
s = s.replace(".jpg", "")
s = s.replace(" ", "")
s = s.replace('(', "")
s = s.replace(')', "")
print s
x = s.split(",")
x = list(map(int, x))
d = {}
for i in range(0, len(x), 2):
	d[x[i]] = x[i+1]

pick = open("labels.pkl", "wb")
pickle.dump(d, pick)
