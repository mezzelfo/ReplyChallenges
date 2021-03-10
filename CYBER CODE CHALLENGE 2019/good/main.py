s = '4VOID_PAN1C_BY_ALL_M3AN$'.lower()
l = list(s)

for i in range(len(s)):
    s = ''.join(l)
    s = s.lower()
    l = list(s)
    l[i] = l[i].upper()
    s = ''.join(l)
    if s[i].isalpha():
        print('{FLG:'+s+'}')
        input()