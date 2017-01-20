import re
str = "My Saurabh name is Saurabh"
words = re.compile("\w+").findall(str)
occurrenceTuplet = []
def sortFn(s):
    return s[1]
for i in words:
    occurrence = words.count(i)
    if ((i, occurrence) not in occurrenceTuplet):
        occurrenceTuplet.append((i, occurrence))
print sorted(occurrenceTuplet, key=sortFn, reverse=True)
