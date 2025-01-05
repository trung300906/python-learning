# first is normal type of number in R zone
a = 10 # int
b = 2.145 # float

# second is complex
c = complex(1, -3) # -3j + 1 
d = -3j + 1
print(c)
print(d)
print(d.imag) # float number
print(d.real) # float number 
print(c.conjugate()) # complex^T
print(type(d), " and ", type(c))

# fractions
from fractions import Fraction # using * for import entire of this lib

e = Fraction(5, 7) # 5/7
f = Fraction(1, 2) # 1/2
# avoid using fraction like Fraction(2,0) => error division to zero
print(e + f)
print(e - f)
print(e * f)
print(e / f)
print(e.limit_denominator()) # 7/5
print(type(e), " ", type(f))

# decimal
from decimal import Decimal, getcontext
getcontext().prec = 20
c = Decimal(30)
d = Decimal(10)
print(d / c)
print(type(c))
print((d * c).sqrt()) #defirence with c++ =))