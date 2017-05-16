import json

def merge(output, id, *args):
	js = []
	for filename in args:
		f = open(filename)
		js.append(json.loads("".join(f.readlines())))
		f.close()
	result = {}
	cpu = []
	for r_cpu in zip(*[x['cpu'] for x in js]):
		agregate = {
			"fw": [x['fw'] for x in r_cpu],
			"sample": [x['sample'] for x in r_cpu],
			"total": [x['total'] for x in r_cpu]
		}
		cpu.append(agregate)
	result['cpu'] = cpu
	gpu = []
	for r_cpu in zip(*[x['gpu'] for x in js]):
		agregate = {
			"fw": [x['fw'] for x in r_cpu],
			"sample": [x['sample'] for x in r_cpu],
			"total": [x['total'] for x in r_cpu]
		}
		gpu.append(agregate)
	result['gpu'] = gpu

	result[id] = js[0][id]

	f = open(output, 'w+')
	f.write(json.dumps(result, sort_keys=True, indent=4, separators=(',', ': ')))
	f.close()



merge('data_merged', 'data_sizes', 'data_comparison_13_05_2017 17:18:38', 'data_comparison_13_05_2017 19:50:02', 'data_comparison_13_05_2017 22:25:18')
merge('sample_merged', 'sample_sizes', 'sample_comparison_13_05_2017 18:47:24','sample_comparison_13_05_2017 21:18:38','sample_comparison_13_05_2017 23:54:01')
merge('kmer_merged', 'kmer_sizes', 'kmer_comparison_13_05_2017 17:41:00','kmer_comparison_13_05_2017 20:12:25','kmer_comparison_13_05_2017 22:47:46')
merge('skip_merged', 'skip_sizes', 'skip_comparison_13_05_2017 18:33:14','skip_comparison_13_05_2017 21:04:30','skip_comparison_13_05_2017 23:39:53')