"""Count words."""
import re
def sortFn(s):
    return str(s[1])+str(sorted(s[0], reverse=True))

def count_words(s, n):
    """Return the n most frequently occuring words in s."""
    wordsOccurArray = []
    wordsArray = []
    top_n = []
    # TODO: Count the number of occurences of each word in s
    wordsArray = re.compile("\w+").findall(s)
    for i in wordsArray:
        occurrence = wordsArray.count(i)
        if ((i, occurrence) not in top_n):
            top_n.append((i, occurrence))
     # TODO: Sort the occurences in descending order (alphabetically in case of ties)
     #Sorted in terms of occurrences
    top_n = sorted(top_n, key=sortFn, reverse=True)
    #Sorted in descending order alphabetically
    i=0
    tmp_top = []
    tmp_comparator = []
    resetComparator = False
    lastElem = (0,0)
    while i < len(top_n):
        if (top_n[i][1] == lastElem[1]):
            if (resetComparator):
                tmp_comparator = []
                tmp_comparator.append(lastElem)
                resetComparator = False
                tmp_top.pop(len(tmp_top) - 1)
            tmp_comparator.append(top_n[i])
            tmp_comparator = sorted(tmp_comparator)
        else:
            tmp_top = tmp_top + tmp_comparator
            tmp_top.append(top_n[i])
            resetComparator = True
        lastElem = top_n[i]
        i = i+1
        if (i >= len(top_n)):
            tmp_top = tmp_top + tmp_comparator
    top_n = tmp_top
    # TODO: Return the top n words as a list of tuples (<word>, <count>)
    
    return top_n[0:n]


def test_run():
    """Test count_words() with some inputs."""
    print count_words("cat bat cat cat bat cat", 3)
    print count_words("betty bought a bit of butter but the butter was bitter", 3)

test_run()
