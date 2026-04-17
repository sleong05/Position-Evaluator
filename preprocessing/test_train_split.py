# split.py
import pandas as pd

print("Splitting into train and test...")

train_file = open('train.csv', 'w')
test_file  = open('test.csv', 'w')

header_written = False
total_train = 0
total_test  = 0

for i, chunk in enumerate(pd.read_csv('chess_data.csv', chunksize=10000)):
    if not header_written:
        chunk.iloc[:0].to_csv(train_file, index=False)
        chunk.iloc[:0].to_csv(test_file, index=False)
        header_written = True

    if i % 10 == 0:
        chunk.to_csv(test_file, index=False, header=False)
        total_test += len(chunk)
    else:
        chunk.to_csv(train_file, index=False, header=False)
        total_train += len(chunk)

    if i % 100 == 0:
        print(f"  processed {i * 10000:,} rows...")

train_file.close()
test_file.close()

print(f"\nDone!")
print(f"train.csv — {total_train:,} rows")
print(f"test.csv  — {total_test:,} rows")