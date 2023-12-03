import re

data = "task: 2149,179,4268,2613,1065,4205,2124,1798 | cost: 350.20133849957347"
match = re.search(r'task: ([\d,]+) \| cost: (\d+\.\d+)', data)
if match:
    x = match.group(1)
    y = match.group(2)
    print("Task:", x)
    print("Cost:", y)
    print(type(x))
else:
    print("Data format not matched.")

