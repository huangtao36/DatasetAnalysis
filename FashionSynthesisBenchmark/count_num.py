import pickle
from collections import Counter

with open('./class_label.pkl', mode='rb') as f:
    label_dic = pickle.load(f)


geneder, style, color, sleeve = list(), list(), list(), list()
for value in label_dic.values():
    geneder.append(value['geneder'])
    style.append(value['style'])
    color.append(value['color'])
    sleeve.append(value['sleeve'])

a = Counter(geneder)
print(f'Geneder: {a}')

b = Counter(style)
print(f'style: {b}')

c = Counter(color)
print(f'color: {c}')

d = Counter(sleeve)
print(f'sleeve: {d}')
