from random import uniform

def random_data(n):
	f = open('_data_len_'+str(n), 'w+')
	for _ in range(n):
		f.write(str(uniform(45.0, 75.0)) + '\n')
	f.close()

for n in [50,100,200,350,500,750,1000,2000,3000,4000,5000,10000,50000,100000,250000,500000,1000000]:
	random_data(n)
