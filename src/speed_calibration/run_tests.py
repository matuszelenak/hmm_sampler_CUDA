import subprocess
import json
import re
import time

data_sizes = [100000,500000,1000000,5000000,10000000,50000000,100000000,500000000,1000000000]

data_size_comparison = {'data_sizes' : data_sizes, 'cpu' : [], 'gpu' : []}
for d in data_sizes:
	bashCommand = "./calibration {0}".format(d)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	output = output.decode("utf-8")
	print(output)
	cpu_res = re.search("CPU ([0-9]+) ms", output).group(1)
	gpu_res = re.search("GPU ([0-9]+) ms", output).group(1)
	data_size_comparison['cpu'] = data_size_comparison['cpu'] + [{"total" : cpu_res}]
	data_size_comparison['gpu'] = data_size_comparison['gpu'] + [{"total" : gpu_res}]

filename = time.strftime("../../../benchmarks_v3/calibration_%d_%m_%Y %H:%M:%S")
f = open(filename, 'w+')
f.write(json.dumps(data_size_comparison, sort_keys=True, indent=4, separators=(',', ': ')))
f.close()