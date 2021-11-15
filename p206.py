a = [1,2,3]
b = [2,3,4]

dot_prod = 0

for alpha, beta in zip(a,b):
    dot_prod += alpha*beta

print(dot_prod)