a = [0, 1, 2]
# with open('test.log', 'w') as f:
#   for e in a:
#     f.write(str(e)+'\n')

with open('test.log', 'r') as f:
  lines = f.readlines()

print(lines)