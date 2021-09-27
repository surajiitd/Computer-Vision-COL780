def add(arr1,arr2):
	n,m = len(arr1),len(arr1[0])
	arr3 = [[0 for x in range(m)] for x in range(n)]
	for i in range(n):
		for j in range(m):
			arr3[i][j] = arr1[i][j] + arr2[i][j]
	return arr3
def sub(arr1,arr2):
	n,m = len(arr1),len(arr1[0])
	arr3 = [[0 for x in range(m)] for x in range(n)]
	for i in range(n):
		for j in range(m):
			arr3[i][j] = arr1[i][j] - arr2[i][j]
	return arr3


arr1 = [[1,2,3], [1,1,1]]
arr2 = [[1,2,3], [1,1,1]]
arr3 = add(arr1,arr2)
arr4 = sub(arr1,arr2)

print(arr3)
print(arr4)