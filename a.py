import spartan as sp
sp.initialize(['--cluster=1','--num_workers=16','--hosts=192.168.1.54:2,192.168.1.55:2,192.168.1.56:2,192.168.1.57:2,192.168.1.58:2,192.168.1.59:2,192.168.1.60:2,192.168.1.61:2'])
print "#1"
a=sp.ones((100,100))
print "#2"
b=sp.ones((100,100))
print "#3"
c=a+b
print c.evaluate().glom()


