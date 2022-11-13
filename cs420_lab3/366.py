import math

res1 = 'F7E75FDC469067FFDC4E847C51F452DF'
p = int(res1, 16)

res2 = 'E85CED54AF57E53E092113E62F436F4F'
q = int(res2, 16)

res3 = '0D88C3'
e = int(res3, 16)

#calculating values of n and f
n = p*q
f = (p-1)*(q-1)

#calculating the two numbers
def calculate(a,b):
        if b==0:
                return a
        else:
                return calculate(b,a%b)


for m in range(2,f):
        if calculate(m,f) == 1:
                break

#calculating value of private key
for i in range(1,10):
        x = 1 + i*f
        if x % m == 0:
                g = int(x/m)
                break

enc = ('39cf8ba4e56530ae0a9d9d6072464fb253d55d4d5fdbd3789d28996a3f877879').encode("utf-8")
enc = int(enc.hex(),16)

print("The encrypted message is: ", enc)

d = int((1/e)%((( 1 + math.sqrt(5) ) / 2)*n)) 

decmsg = pow(enc,d,n)
decmsg = hex(decmsg)
decmsg = bytes.fromhex(decmsg).decode("utf-8")

print("the decrypted message is:", decmsg)
