inFile = open(r'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\BackupData\UncleanedData\uncleaned_test_competition_ratios1-4.csv', 'r')

outFile = open(r'E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\BackupData\CleanedData\cleaned_test_competition_ratios1-4.csv', 'w')

listLines = []

for line in inFile:
    line = line.replace("[", "")
    line = line.replace("]", "")
    line = line.replace("'", "")
    line = line.replace("\"", "")
    line = line.replace(" ", "")
    print(line)
    outFile.write(line)

outFile.close()
inFile.close()