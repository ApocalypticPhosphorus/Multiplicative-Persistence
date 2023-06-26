import threading
import time

import numpy as np
import matplotlib.pyplot as plt

def persistence(x):
    persistence_count = 0
    
    while x >= 10:
        product = 1

        # Calculate the product of the digits
        #print(str(x) + str(prime_factors(x)))
        while x > 0:
            x, digit = divmod(x, 10)
            product *= digit

        x = product
        persistence_count += 1

    return persistence_count

def greatest_zero_position(number):
    number_str = str(number)
    max_position = -1

    for i, digit in enumerate(number_str):
        if digit == '0':
            max_position = max(max_position, len(number_str) - i)

    return max_position


def check_numbers(start, target):
    number = start
    while persistence(number) < target:
        number += 4
        if greatest_zero_position(number) != -1:
            number += pow(10, greatest_zero_position(number) - 1)
        print(f"Checking: {number}", end='\r')
    print("\n")
    print(number)

"""target = int(input("target: "))
n_threads = 4  # Number of threads you want to use

start_time = time.perf_counter()
threads = []
for i in range(n_threads):
    start_number = 1000000 + i  # Starting number for each thread
    t = threading.Thread(target=check_numbers, args=(start_number, target))
    t.start()
    threads.append(t)

# Wait for all threads to finish
for t in threads:
    t.join()
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Generate x values
x_values = np.arange(1, 1000)

# Calculate persistence for each x value
persistence_values = np.array([persistence(x) for x in x_values])"""

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


"""prim = [0, 10, 25, 39, 77, 679, 6788, 68889, 2677889, 26888999, 3778888999, 277777788888899]
for p in prim:
    persistence(p)"""
"""
for x in range(300,1000):
    for y in range(100):
        for z in range(100):
            print(f"Checking x: {x} y: {y} z: {z}", end='\r')
            num = pow(2,x)*pow(3,y)*pow(7,z)
            if persistence(num) == 11:
                print("\n")
                print(f"Number: {num}, x: {x} y: {y} z: {z}")"""

import multiprocessing
import sys

def find_persistence(x_range, y_range, z_range):
    for x in x_range:
        for y in y_range:
            for z in z_range:
                num = pow(2, x) * pow(3, y) * pow(7, z)
                if persistence(num) == 11:
                    print("\n")
                    print(f"Number: {num}, x: {x}, y: {y}, z: {z}")
start_time = time.perf_counter()

if __name__ == '__main__':
    num_processes = multiprocessing.cpu_count()  # Number of CPU cores
    x_range = range(1000)
    y_range = range(1000)
    z_range = range(1000)

    progress_queue = multiprocessing.Queue()

    processes = []
    for i in range(num_processes):
        x_subset = x_range[i::num_processes]  # Split x_range into subsets
        p = multiprocessing.Process(target=find_persistence, args=(x_subset, y_range, z_range))
        processes.append(p)
        p.start()
    
    # Print progress messages from the queue
    while any(p.is_alive() for p in processes):
        while not progress_queue.empty():
            message = progress_queue.get()
            print(message, end='\r')

    for p in processes:
        p.join()
        
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

#print(persistence(323891164381446144))
"""
# Generate x values
x_values = np.arange(1, 10000)

# Calculate persistence for each x value
persistence_values = np.array([persistence(x) for x in x_values])
prime_values = np.array([prime_factors(x) for x in x_values])

# Plot the results
plt.plot(x_values, persistence_values, color='blue', label='Line 1')
plt.plot(x_values, prime_values, color='red', label='Line 2')
plt.xlabel('x')
plt.ylabel('Persistence')
plt.title('Multiplicative Persistence')
plt.grid(True)
plt.show()"""