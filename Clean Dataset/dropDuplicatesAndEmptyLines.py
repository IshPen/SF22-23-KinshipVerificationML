inFile = open('cleaned_test_competition.csv','r')

outFile = open('cleaned_test_competition2.csv', 'w')

listLines = []

for line in inFile:

    if line in listLines:
        continue

    else:
        outFile.write(line)
        listLines.append(line)

outFile.close()

inFile.close()