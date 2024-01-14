x= 1
y = 0.1
z= 1

for i in range(10000):
    print(f"{x/y}  |  {x/z}")
    y = y - (y/y**2)
    z = z - (z/z**2)
