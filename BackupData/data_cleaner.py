#remove the first two and last two characters of each line in the csv file
#and write the result to a new file
import os

original_csv_root = r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\BackupData\UncleanedData"
original_csv = r"uncleaned_test_competition_ratios1.csv"


new_csv_root = r"E:\Programs\Program Files\Pycharm Projects\SF22-23-KinshipVerificationML\BackupData\CleanedData"
new_csv = os.path.join(new_csv_root, original_csv)

original_csv = os.path.join(original_csv_root, original_csv)

def main():
    with open(original_csv, 'r') as f:
        with open(new_csv, 'w') as f_out:
            for line in f:
                line = line.strip()
                line = line[2:-2]
                f_out.write(line + '\n')
    print('Done')
#remove all duplicate lines in the csv file
def remove_duplicates():
    inFile = open(new_csv, 'r')
    outFile = open(new_csv, 'w')
    listLines = []

    for line in inFile:
        if line in listLines:
            continue
        else:
            outFile.write(line)
            listLines.append(line)
    outFile.close()
    inFile.close()

if __name__ == '__main__':
    main()
    #remove_duplicates()
    print('Done')