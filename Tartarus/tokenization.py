import re

with open("../data/content.txt", "r") as f:
    content = f.read()
print(content[0:100])
chars = sorted(list(set(content)))

print(chars[0:3])
