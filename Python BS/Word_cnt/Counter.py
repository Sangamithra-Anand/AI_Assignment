from collections import Counter

text = "This is a sample text. This text will be used to demonstrate the word counter."
words = text.split()
word_count = Counter(words)

print(word_count)
