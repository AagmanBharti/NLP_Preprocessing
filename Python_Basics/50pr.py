for n in range(50, 1, -1):
    for i in range(2, n):
        if n % i == 0:
            break
    else:
        print(n)
        break