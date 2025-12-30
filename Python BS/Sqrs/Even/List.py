# Method 3: Even numbers using list comprehension
even_squares = [num**2 for num in range(100, 201) if num % 2 == 0]
print(even_squares)
