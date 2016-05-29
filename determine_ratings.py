f = open('rated_posts.csv', 'r')
out = open('final_rated_posts.csv', 'w')
out_2 = open('only_ratings', 'w')

counts = dict()
counts[1] = 0
counts[2] = 0
counts[3] = 0
counts[4] = 0
counts[5] = 0
for line in f:
	splits = line.split('@@')
	new_line = splits[0] + '@@' + splits[1] + '@@'
	if len(splits) == 4 and int(splits[3]) != 0:
		value = int(int(splits[2])/int(splits[3]))
		new_line += str(value) + '\n'
		out_2.write(str(value) + '\n')
		counts[value] += 1
	else:
		new_line += str(0) + '\n'
		out_2.write(str(0) + '\n')
	out.write(new_line)

f.close()
out.close()
out_2.close()

for i in counts:
	print str(i) + "\t" + str(counts[i])