a = 10
b = 20
c = a + b
print(a, b , c)

c = a- b
print(c)

c = a * b
print(c)

c = a / b
print(c)

c = a % b #same with c++
print(c)

c = a ** b # same as a^b
print(c)

c = a // b #make real mumber of this conquerse
print(c)

## with float numbers ##
a = 0.1
b = 0.2
c = a + b
print(c)
#for more make sure uisng if
if c == 0.3:
    print("True")
else:
    print("False")

##########
b += 2
b -= 1
b *= 3
b /= 2
b //= 2
c = 2
c **= b
print(c)
print(b)
# about := 
# b = (b+=2 ) - 2 #error traceback
b = (b:= b +2) -(c:= c+4)
print(b)
