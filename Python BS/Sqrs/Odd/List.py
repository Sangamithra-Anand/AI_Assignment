# Method 3: Odd numbers using list comprehension
odd_squares = [num**2 for num in range(100, 201) if num % 2 != 0]
print(odd_squares)
