n = int(input("Enter a number: "))
count = 0
a = 2
if n <= 1:
    print("Not prime.")
while True:
    flag = True
    for i in range(2,a):
        if a % i == 0:
            flag = False
            break
    if flag:
        count += 1
    if count == n:
            print(f"The {n}th prime number is: {a}")
            break
    a += 1
