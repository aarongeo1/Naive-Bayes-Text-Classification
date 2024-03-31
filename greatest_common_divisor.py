flowerbed = [1,0,0,0,0,1]
n = 2

n1 =0
for i in range(0,len(flowerbed)-3):
    if flowerbed[i+1] == 0 and flowerbed[i+2] == 0 and flowerbed[i] == 0:
        n1+=1
        i+=3
print(n1)


