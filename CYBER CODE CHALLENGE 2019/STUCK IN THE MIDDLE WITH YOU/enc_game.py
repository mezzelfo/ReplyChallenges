from hashlib import md5
from binascii import hexlify, unhexlify

key= "" #feed_me 
BLK = 16
RND = 3
s = [
		   99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118,
		   202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192,
		   183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21,
		   4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117,
		   9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132,
		   83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207,
		   208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168,
		   81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210,
		   205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115,
		   96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219,
		   224, 50, 58, 10, 73, 6, 36, 92,194, 211, 172, 98, 145, 149, 228, 121,
		   231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8,
		   186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138,
		    112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158,
		   225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223,
		   140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22
]

permutation = [14, 13, 12, 11, 10, 0, 1, 2, 3, 4, 9, 8, 7, 6, 5, 15]

def main():
    mykey = 16
    print(mykey)
    mytext = b"StuckInTheMiddle"
    print("ecco:",enc(mykey,mytext))

    print("My Battery Is Low and It's Getting Dark...\n")
    print("StuckInTheMiddle: b3783a044b06141859d65a1e6fdbfc44")
    print("flag: {FLG:65defb7f4c29b989a47a2dacf8e5efcf}")

def xor(a, b):
    
    x = bytearray(len(a)) 
    for i in range(len(a)):
        x[i] = a[i] ^ b[i]
    return x

def enc(key, plaintext):
    perm_cipher = bytearray(BLK)
    #key = bytearray(unhexlify(md5(key).hexdigest()))
    key = bytearray(key)
    ciphertext = bytearray(plaintext)
    # 3round-encryption
    for _ in range(RND):
        ciphertext = xor(ciphertext, key)
        for i in range(BLK):
            ciphertext[i] = s[ciphertext[i]]
        for i in range(BLK):
            perm_cipher[i] = ciphertext[permutation[i]]
        ciphertext = perm_cipher
    return hexlify(ciphertext)

def double_enc(key, plaintext):
    assert len(key) == 8
    key1 = key[:4]
    key2 = key[4:]
    ciphertext_p1 = unhexlify(enc(key1, plaintext))
    ciphertext = enc(key2, ciphertext_p1)
    return ciphertext


if __name__ == "__main__":
    main()
		


