import random

a = random.randint(1, 10)
b = random.randint(1, 10)

correct = a * b

while True:
    user_ans = int(input(f"What is {a} x {b}? "))

    if user_ans == correct:
        print("Correct!")
        break
    else:
        print("Wrong! Try again.")
