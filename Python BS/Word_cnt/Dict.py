text = "This is a sample text. This text will be used to demonstrate the word counter."
words = text.split()  # Split text into words
word_count = {}

for word in words:
    if word in word_count:
        word_count[word] += 1  # If word exists, increase count by 1
    else:
        word_count[word] = 1   # If word is new, set count to 1

print(word_count)
