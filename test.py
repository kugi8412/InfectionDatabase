import timeit
setup_code = '''
existing_list = list(range(3))
def new_value(x):
    return x * 2
'''

# Tworzenie nowej listy
new_list_time = timeit.timeit('new_list = [new_value(item) for item in existing_list]', setup=setup_code, number=100000)

# Modyfikacja na miejscu
in_place_time = timeit.timeit('for i in range(len(existing_list)): existing_list[i] = new_value(existing_list[i])', setup=setup_code, number=100000)

print(f'Tworzenie nowej listy: {new_list_time} sekundy')
print(f'Modyfikacja na miejscu: {in_place_time} sekundy')

def test(t):
    t[1] = 0
    return t



array = [1, 2, 3]
print(array)
a = test(array)
print(array)
