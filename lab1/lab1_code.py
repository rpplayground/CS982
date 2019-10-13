#%% [markdown]
## University of Strathclyde -  MSc Artificial Intelligence and Applications
## CS982 - Big Data Technologies
### Lab 1
# File Created first created 2nd October 2019 by Barry Smart.
# 
#### ABOUT:
# This file walks through the examples set out in the Lab1.pdf.

#%%
# 1. Write a Python program to get the Python version you are using and print it out

import platform
print(platform.sys.version)

import sys
print(sys.version_info)

#%%
# 2. Write a Python program to print the following as shown:

string_to_print = \
"Baa, baa, black sheep\n \
\t\tHave you any wool?\n \
\tYes sir, yes sir\n \
\t\t\tThree bags full.\n \
One for my master\n \
\tAnd one for the dame\n \
One for the little boy\n \
Who lives down the lane."

print(string_to_print)

#%%
# 3. Write a Python program to count the number of even and odd numbers from a series of numbers

def count_odd_and_even_numbers_in_list(list_of_numbers):
    length_of_list = len(list_of_numbers)
    even_count = 0
    odd_count = 1
    for count in range (0, length_of_list - 1):
        if list_of_numbers[count] % 2 == 0:
            # The number is even, so increment the even count by one
            even_count = even_count + 1
        else:
            # The number is odd, so increment the odd count by one
            odd_count = odd_count + 1
    return even_count, odd_count

even, odd = count_odd_and_even_numbers_in_list([1, 2, 3, 4, 12, 32, 27, 5, 6, 7, 8, 9])
print("Even count:", even, ", odd count:", odd)

#%%
# 4. Write a Python program that prints all the numbers from 0 to 50 except 37 and 16.

def capture_numbers_in_range_except(maximum, list_of_numbers_to_exclude):
    list_of_numbers = []
    for number in range (0, maximum + 1):
        if number not in list_of_numbers_to_exclude:
            list_of_numbers.append(number)
    return list_of_numbers

capture_numbers_in_range_except(50, [37,16])

#%%
# 5. Write a Python program to get the Fibonacci series between 0 and 100. The Fibonacci Sequence is the series of numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ... The next number is found by adding up the two numbers before it.

def create_finonacci(maximum):
    fibonacci_sequence_list = [0]
    next_number_in_sequence = 1
    while next_number_in_sequence < maximum:
        # Append number to the list
        fibonacci_sequence_list.append(next_number_in_sequence)
        # Check length of sequence
        length_of_sequence = len(fibonacci_sequence_list)
        # Calculate next candidate number in sequence
        next_number_in_sequence = fibonacci_sequence_list[length_of_sequence - 2] + fibonacci_sequence_list[length_of_sequence - 1]
    return fibonacci_sequence_list

create_finonacci(100)
