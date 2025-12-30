import random

num1 = random.randint(1, 10)
num2 = random.randint(1, 10)

print("Multiply:", num1, "x", num2)
user_answer = int(input("Enter the product: "))

if user_answer == num1 * num2:
    print("Correct!")
else:
    print("Wrong! The correct answer is:", num1 * num2)
