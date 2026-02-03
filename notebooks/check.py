import csv

max_len = 0
max_value = ""
rows_with_long_fields = 0

with open("data/parallel/en-war/en-war.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        has_long_field = False
        for field in row:
            if len(str(field)) > 512:
                has_long_field = True
        if has_long_field:
            rows_with_long_fields += 1

print(f"\nRows with fields > 512 characters: {rows_with_long_fields}")
