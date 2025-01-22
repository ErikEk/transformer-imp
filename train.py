with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
