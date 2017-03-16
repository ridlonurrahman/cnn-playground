def is_prime(n):
    return all([(n%j) for j in range(2, int(n**0.5)+1)]) and n>1

total = 0
for i in range(86226508):
    if is_prime(i):
        total += i
print total