import string

text = "This is a sample text. This text will be used to demonstrate the word counter."
text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
words = text.split()
word_count = {}

for word in words:
    word_count[word] = word_count.get(word, 0) + 1

print(word_count)
