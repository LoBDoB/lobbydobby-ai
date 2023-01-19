from d_knn import lectureKNN


knn = lectureKNN()

for i in range(20):
    exist_result = knn.existUser(f'user{i+1}')
    print(f'user{i+1}')
    print(exist_result, '\n')

new = {'user101':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 5, 0, 0, 0]}
new_result = knn.newUser(new)
print('\n', new_result)