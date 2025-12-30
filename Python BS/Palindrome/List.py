def is_palindrome(text):
    cleaned = ''.join(filter(str.isalnum, text)).lower()
    return cleaned == cleaned[::-1]

# User input
user_text = input("Enter a text: ")
print(is_palindrome(user_text))
