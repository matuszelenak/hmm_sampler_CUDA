from random import uniform

def suffix(n):
	if (n == 0):
		return [""]
	else:
		return [c + s for c in ['A', 'C', 'G', 'T'] for s in suffix(n-1)]

def random_hmm(kmer_size):
	f = open('hmm'+str(kmer_size), 'w+')
	for kmer in suffix(kmer_size):
		f.write(kmer + "\t" + str(uniform(0, 1)) + '\t' + str(uniform(0, 1)) + '\n')
	f.close()

for k in range(1,9):
	random_hmm(k)