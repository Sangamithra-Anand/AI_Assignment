def is_palindrome_recursive(text):
    # Clean once
    cleaned = ''.join(c.lower() for c in text if c.isalnum())

    def helper(left, right):
        if left >= right:
            return True
        if cleaned[left] != cleaned[right]:
            return False
        return helper(left + 1, right - 1)

    return helper(0, len(cleaned) - 1)

s = input("Enter text: ")
print(is_palindrome_recursive(s))
