import subprocess
import re
import json
import time
from random import uniform

def suffix(n):
	if (n == 0):
		return [""]
	else:
		return [c + s for c in ['A', 'C', 'G', 'T'] for s in suffix(n-1)]

def random_hmm(kmer_size):
	f = open('dataset/hmm'+str(kmer_size), 'w+')
	for kmer in suffix(kmer_size):
		f.write(kmer + "\t" + str(uniform(0, 1)) + '\t' + str(uniform(0,1)) + '\n')
	f.close()

def random_data(n):
	f = open('dataset/_data_len_'+str(n), 'w+')
	for _ in range(n):
		f.write(str(uniform(0,1)) + '\n')
	f.close()

def run_test(kmer, data, skip, sample, method):
	if (4**kmer * data * 8 + sample * data * 4) > 8 * (1024**3):
		return {"error" : "ram"}
	r = {}
	bashCommand = "./bachelor_thesis/src/hmm_sampler --model dataset/hmm{0} --raw-input --scale dataset/_data_len_{1} --method {2} --sample {3} --maxskip {4}".format(kmer,data,method,sample,skip)
	print (bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	output = output.decode("utf-8")
	print(output)
	if (re.search("FW MATRIX .+ TOOK ([0-9]+) ms", output) != None):
		r['fw'] = re.search("FW MATRIX .+ TOOK ([0-9]+) ms", output).group(1)
	if (re.search("SAMPLING .+ TOOK ([0-9]+) ms", output) != None):
		r['sample'] = re.search("SAMPLING .+ TOOK ([0-9]+) ms", output).group(1)
	if re.search("SAMPLING TOTAL TOOK ([0-9]+) ms", output):
		r['total'] = re.search("SAMPLING TOTAL TOOK ([0-9]+) ms", output).group(1)
	if re.search("DECODE .+ TOOK ([0-9]+) ms", output):
		r['decode'] = re.search("DECODE .+ TOOK ([0-9]+) ms", output).group(1)
	if int(r['total']) > 15*60*1000:
		r['error'] = 'timeout'
	return r

def run_comparisons():
	results = {}
	data_sizes = [50,100,250,500,750,1000,1250,1500,2000,3000,4000,5000,7500,10000,15000,20000,35000,50000,100000,250000,500000,1000000]
	kmer_sizes = [x + 1 for x in range(8)]
	skip_sizes = [x + 1 for x in range(8)]
	sample_sizes = [5,10,25,50,75,100,150,200,500,1000,1500,3000,5000]

	for data_size in data_sizes:
		random_data(data_size)
	for kmer in kmer_sizes:
		random_hmm(kmer)

	print("Data generation done")

	#performance depending on data size. Default kmer size 6, skip size 1, sample size 100
	kmer = 6
	skip = 1
	sample = 100
	data_size_comparison = {'data_sizes' : data_sizes, 'cpu' : [], 'gpu' : []}
	for data_size in data_sizes:
		print("Running on GPU for data ",data_size)
		res_gpu = run_test(kmer, data_size, skip, sample, 'GPU')
		if ('error' in res_gpu):
			break
		data_size_comparison['gpu'] = data_size_comparison['gpu'] + [{'fw' : res_gpu['fw'], 'sample' : res_gpu['sample'], 'total' : res_gpu['total']}]
	for data_size in data_sizes:
		print("Running on CPU for data ",data_size)
		res_cpu = run_test(kmer, data_size, skip, sample, 'CPU')
		if ('error' in res_cpu):
			break
		data_size_comparison['cpu'] = data_size_comparison['cpu'] + [{'fw' : res_cpu['fw'], 'sample' : res_cpu['sample'], 'total' : res_cpu['total']}]
	filename = time.strftime("benchmarks_v3/data_comparison_%d_%m_%Y %H:%M:%S")
	f = open(filename, 'w+')
	f.write(json.dumps(data_size_comparison, sort_keys=True, indent=4, separators=(',', ': ')))
	f.close()


	#performance depending on HMM size
	data = 5000
	skip = 1
	sample = 100
	kmer_size_comparison = {'kmer_sizes' : kmer_sizes, 'cpu' : [], 'gpu' : []}
	for kmer_size in kmer_sizes:
		print("Running on GPU for kmer ",kmer_size)
		res_gpu = run_test(kmer_size, data, skip, sample, 'GPU')
		if ('error' in res_gpu):
			break
		kmer_size_comparison['gpu'] = kmer_size_comparison['gpu'] + [{'fw' : res_gpu['fw'], 'sample' : res_gpu['sample'], 'total' : res_gpu['total']}]
	for kmer_size in kmer_sizes:
		print("Running on CPU for kmer ",kmer_size)
		res_cpu = run_test(kmer_size, data, skip, sample, 'CPU')
		if ('error' in res_cpu):
			break
		kmer_size_comparison['cpu'] = kmer_size_comparison['cpu'] + [{'fw' : res_cpu['fw'], 'sample' : res_cpu['sample'], 'total' : res_cpu['total']}]
	filename = time.strftime("benchmarks_v3/kmer_comparison_%d_%m_%Y %H:%M:%S")
	f = open(filename, 'w+')
	f.write(json.dumps(kmer_size_comparison, sort_keys=True, indent=4, separators=(',', ': ')))
	f.close()

	#performance depending on skip size
	#default kmer size 6, data 3000 sample 100, 
	data = 3000
	sample = 100
	kmer = 6
	skip_size_comparison = {'skip_sizes' : [x+1 for x in range(kmer)], 'cpu' : [], 'gpu' : []}
	for	skip_size in range(1,7):
		print("Running on GPU for skip ",skip_size)
		res_gpu = run_test(kmer, data, skip_size, sample, 'GPU')
		if ('error' in res_gpu):
			break
		skip_size_comparison['gpu'] = skip_size_comparison['gpu'] + [{'fw' : res_gpu['fw'], 'sample' : res_gpu['sample'], 'total' : res_gpu['total']}]
	for	skip_size in range(1,7):
		print("Running on CPU for skip ",skip_size)
		res_cpu = run_test(kmer, data, skip_size, sample, 'CPU')
		if ('error' in res_cpu):
			break
		skip_size_comparison['cpu'] = skip_size_comparison['cpu'] + [{'fw' : res_cpu['fw'], 'sample' : res_cpu['sample'], 'total' : res_cpu['total']}]
	filename = time.strftime("benchmarks_v3/skip_comparison_%d_%m_%Y %H:%M:%S")
	f = open(filename, 'w+')
	f.write(json.dumps(skip_size_comparison, sort_keys=True, indent=4, separators=(',', ': ')))
	f.close()

	#performance depending on sample size
	#default kmer size 6, data 1000, skip 1 
	data = 3000
	skip = 1
	kmer = 6
	sample_size_comparison = {'sample_sizes' : sample_sizes, 'cpu' : [], 'gpu' : []}
	for	sample_size in sample_sizes:
		print("Running on GPU for sample ",sample_size)
		res_gpu = run_test(kmer, data, skip, sample_size, 'GPU')
		if ('error' in res_gpu):
			break
		sample_size_comparison['gpu'] = sample_size_comparison['gpu'] + [{'fw' : res_gpu['fw'], 'sample' : res_gpu['sample'], 'total' : res_gpu['total']}]
	for	sample_size in sample_sizes:
		print("Running on CPU for sample ",sample_size)
		res_cpu = run_test(kmer, data, skip, sample_size, 'CPU')
		if ('error' in res_cpu):
			break
		sample_size_comparison['cpu'] = sample_size_comparison['cpu'] + [{'fw' : res_cpu['fw'], 'sample' : res_cpu['sample'], 'total' : res_cpu['total']}]
	filename = time.strftime("benchmarks_v3/sample_comparison_%d_%m_%Y %H:%M:%S")
	f = open(filename, 'w+')
	f.write(json.dumps(sample_size_comparison, sort_keys=True, indent=4, separators=(',', ': ')))
	f.close()

	#performance depending on sample size
	#default kmer size 6, data 1000, skip 1 
	skip = 1
	kmer = 6
	samplexdata = [(data, sample) for data in data_sizes[:8] for sample in sample_sizes[3:]]
	decode_comparison = {'input_sizes' : [x*y for x,y in samplexdata], 'cpu' : [], 'gpu' : []}
	for	data, sample_size in samplexdata:
		print("Running on GPU for sample ",sample_size)
		res_gpu = run_test(kmer, data, skip, sample_size, 'GPU')
		if ('error' in res_gpu):
			break
		decode_comparison['gpu'] = decode_comparison['gpu'] + [{'decode' : res_gpu['decode']}]
	for	data, sample_size in samplexdata:
		print("Running on CPU for sample ",sample_size)
		res_cpu = run_test(kmer, data, skip, sample_size, 'CPU')
		if ('error' in res_cpu):
			break
		decode_comparison['cpu'] = decode_comparison['cpu'] + [{'decode' : res_cpu['decode']}]
	filename = time.strftime("benchmarks_v3/decode_comparison_%d_%m_%Y %H:%M:%S")
	f = open(filename, 'w+')
	f.write(json.dumps(decode_comparison, sort_keys=True, indent=4, separators=(',', ': ')))
	f.close()


run_comparisons()
