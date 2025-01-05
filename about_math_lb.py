import math
a = 3.5
print(math.trunc(a)) # trả về phần nguyên
print(math.floor(a)) # làm tròn xuống
print(math.ceil(a)) # làm tròn lên
print(math.fabs(-a)) # lấy trị tuyệt đối
print(math.sqrt(a)) # lấy căn
print(math.gcd(math.ceil(a), 10)) # lấy gcd của phần làm tròn lên của a(là 4) với 10

# about // and trunc

a = 10
b = -3
print(a/b)
print(a//b)
print(math.trunc(a/b))

#Toán tử //: Phép chia lấy phần nguyên sẽ trả về phần nguyên của kết quả chia và làm tròn xuống (floor division). Điều này có nghĩa là kết quả luôn hướng về số nguyên nhỏ hơn hoặc bằng kết quả thực của phép chia.

#Hàm math.trunc(): Hàm này sẽ cắt bỏ phần thập phân và trả về phần nguyên của số. Nó không làm tròn xuống hay lên mà chỉ đơn giản là lấy phần nguyên của số.

# chỉ ó hiệu quả với số