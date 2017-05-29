# Levenshtein Distance
def levenshteinDistance(str1, str2):
    len1 = len(str1)
    len2 = len(str2)

    matrix = [[0 for x in range(len2 + 1)] for y in range(len1 + 1)]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + cost)

    print matrix
    return matrix[len1][len2]


print levenshteinDistance("stecai1", "stecai");
print levenshteinDistance("kitten", "sitting");
print levenshteinDistance("Saturday", "Sunday");
