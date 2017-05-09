from random import uniform

kmer_size = 4

def suffix(n):
	if (n == 0):
		return [""]
	else:
		return [c + s for c in ['A', 'C', 'G', 'T'] for s in suffix(n-1)]

f = open('hmm'+str(kmer_size), 'w+')
for kmer in suffix(kmer_size):
	f.write(kmer + "\t" + str(uniform(45.0, 75.0)) + '\t' + str(uniform(1.0, 2.5)) + '\n')

f.close()
