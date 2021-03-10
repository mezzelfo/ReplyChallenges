enc = [52, 54, 60, 40, 72, 64, 42, 35, 93, 26, 38, 110, 3, 47, 56, 26, 64, 1, 49, 33, 71, 38, 7, 25, 20, 92, 1, 9]
print(len(enc))
flg = list('{FLG:')
for i in range(len(flg)):
    print(chr(ord(flg[i])^enc[i]))


passwd = 'Opportunity'
s = ''.join([chr(ord(passwd[i % len(passwd)])^enc[i]) for i in range(0,27)])
if '{FLG:' in s:
    print(s)
    print(passwd)
        