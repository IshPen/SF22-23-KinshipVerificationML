inFile = open(r'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\BackupData\CleanedData\cleaned_test_competition_ratios1-4.csv', 'r')

outFile = open(r'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\BackupData\CleanedData\cleaned_test_competition_ratios1-4(1).csv', 'w')

listLines = []

for line in inFile:
    if line in listLines:
        continue
    else:
        outFile.write(line)
        listLines.append(line)

outFile.close()

inFile.close()