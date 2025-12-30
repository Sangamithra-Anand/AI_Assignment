import random

for i in range(5):  # Ask 5 questions
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)

    print(num1, "x", num2)
    user_answer = int(input("Enter product: "))

    if user_answer == num1 * num2:
        print("Correct!\n")
    else:
        print("Wrong! Correct answer:", num1 * num2, "\n")
