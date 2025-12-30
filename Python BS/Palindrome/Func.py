def is_palindrome(text):
    cleaned = ""
    for char in text:
        if char.isalnum():
            cleaned += char.lower()
    return cleaned == cleaned[::-1]

# User input
user_text = input("Enter a text: ")
print(is_palindrome(user_text))
