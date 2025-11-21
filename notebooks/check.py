import csv

max_len = 0
max_value = ""

with open("data/parallel/en-ceb/en-ceb.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        for field in row:
            if len(field) > max_len:
                max_len = len(field)
                max_value = field

print("Max length:", max_len)
print("Field value:", max_value)
