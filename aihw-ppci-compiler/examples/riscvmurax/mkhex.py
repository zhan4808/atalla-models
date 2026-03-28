from sys import argv

with open(argv[1] + ".bin", "rb") as f:
    bindata = f.read()

assert len(bindata) % 4 == 0

f0 = open(argv[1] + ".v_toplevel_system_ram_ram_symbol0" + ".bin", "w")
f1 = open(argv[1] + ".v_toplevel_system_ram_ram_symbol1" + ".bin", "w")
f2 = open(argv[1] + ".v_toplevel_system_ram_ram_symbol2" + ".bin", "w")
f3 = open(argv[1] + ".v_toplevel_system_ram_ram_symbol3" + ".bin", "w")
for i in range(len(bindata) // 4):
    w = bindata[4 * i : 4 * i + 4]
    print(f"{w[0]:08b}", file=f0)
    print(f"{w[1]:08b}", file=f1)
    print(f"{w[2]:08b}", file=f2)
    print(f"{w[3]:08b}", file=f3)
f0.close()
f1.close()
f2.close()
f3.close()
