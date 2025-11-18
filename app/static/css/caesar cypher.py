def caesar_cipher_encrypt(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base + shift) % 26 + base)
        else:
            result += char
    return result

plaintext = input("Enter the plaintext: ")
shift_key = int(input("Enter the shift key: "))
ciphertext = caesar_cipher_encrypt(plaintext, shift_key)
print("Encrypted Text:", ciphertext)
