import matplotlib.pyplot as plt
import numpy as np
import json
from numpy import polyfit
import math

def make_graph(x_data, y_cpu, y_gpu, title, xlabel, ylabel, filename):

	z_cpu = np.polyfit(np.array(x_data[:len(y_cpu)]),np.array(y_cpu), len(x_data))
	z_gpu = np.polyfit(np.array(x_data[:len(y_gpu)]),np.array(y_gpu), len(x_data))
	p_cpu = np.poly1d(z_cpu)
	p_gpu = np.poly1d(z_gpu)

	plt.title(title)
	plt.yscale('log')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x_data[:len(y_cpu)], p_cpu(np.array(x_data[:len(y_cpu)])), 'b--', label='CPU')
	plt.plot(x_data[:len(y_gpu)], p_gpu(np.array(x_data[:len(y_gpu)])), 'g--', label='GPU')
	plt.plot(x_data[:len(y_cpu)], y_cpu, 'x')
	plt.plot(x_data[:len(y_gpu)], y_gpu, 'x')
	plt.legend(loc='best')
	plt.savefig(filename + '.png')
	#plt.show()
	plt.clf()
	plt.cla()
	plt.close()

def plot_speedup(x_data, y_cpu, y_gpu, title, xlabel, ylabel, filename, p):
	speedup_y = [cpu/gpu for cpu,gpu in zip(y_cpu, y_gpu)]
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x_data[:len(y_cpu)], speedup_y, '.')
	if p:
		z_speedup = np.polyfit(np.array(x_data[:len(y_cpu)]),np.array(speedup_y), len(x_data))
		p_cpu = np.poly1d(z_speedup)
		plt.plot(x_data[:len(y_cpu)], p_cpu(np.array(x_data[:len(speedup_y)])), 'r-', label='Speedup')

	plt.legend(loc='best')
	plt.savefig(filename + '.png')
	#plt.show()
	plt.clf()
	plt.cla()
	plt.close()

f = open('benchmarks_v3/data_merged')
j = json.loads("".join(f.readlines()))

"""
x_data = j['data_sizes']
y_data_cpu = [sum([int(y) for y in x['fw']])/len(x['fw']) for x in j['cpu']]
y_data_gpu = [sum([int(y) for y in x['fw']])/len(x['fw']) for x in j['gpu']]

make_graph(x_data, y_data_cpu, y_data_gpu, "Execution time for varying data size", "Data size", "Time in ms", "DataCompare")

plot_speedup(x_data, y_data_cpu, y_data_gpu, "Speedup for varying data size", "Data size", "CPU/GPU", "DataCompareSpeedup", False	)

x_data = j['data_sizes']
y_data_cpu = [sum([int(y) for y in x['sample']])/len(x['sample']) for x in j['cpu']]
y_data_gpu = [sum([int(y) for y in x['sample']])/len(x['sample']) for x in j['gpu']]

make_graph(x_data, y_data_cpu, y_data_gpu, "Execution time for varying data size", "Data size", "Time in ms", "DataSampleCompare")

plot_speedup(x_data, y_data_cpu, y_data_gpu, "Speedup for varying data size", "Data size", "CPU/GPU", "DataSampleCompareSpeedup", True)


f = open('benchmarks_v3/sample_merged')
j = json.loads("".join(f.readlines()))

x_data = j['sample_sizes']
y_data_cpu = [sum([int(y) for y in x['sample']])/len(x['sample']) for x in j['cpu']]
y_data_gpu = [sum([int(y) for y in x['sample']])/len(x['sample']) for x in j['gpu']]

#print(" & ".join(map(str, x_data[:len(y_data_gpu)])) + "\\\\")
#print(" & ".join(list(map(str, y_data_cpu)) + ['N/A' for _ in range(len(y_data_gpu) - len(y_data_cpu))]) + "\\\\")
#print(" & ".join(list(map(str, y_data_gpu))) + "\\\\")
#print(" & ".join([str(math.floor(x/y)) for x,y in zip(y_data_cpu, y_data_gpu)] + ['N/A' for _ in range(len(y_data_gpu) - len(y_data_cpu))]) + "\\\\")

make_graph(x_data, y_data_cpu, y_data_gpu, "Execution time for number of samples", "Number of samples", "Time in ms", "SampleCompare")

plot_speedup(x_data, y_data_cpu, y_data_gpu, "Speedup for varying number of samples", "Number of samples", "CPU/GPU", "SampleCompareSpeedup", True)


f = open('benchmarks_v3/skip_merged')
j = json.loads("".join(f.readlines()))

x_data = j['skip_sizes']
y_data_cpu = [sum([int(y) for y in x['fw']])/len(x['fw']) for x in j['cpu']]
y_data_gpu = [sum([int(y) for y in x['fw']])/len(x['fw']) for x in j['gpu']]

make_graph(x_data, y_data_cpu, y_data_gpu, "Execution time for maximum skip", "Skips allowed", "Time in ms", "SkipCompare")

plot_speedup(x_data, y_data_cpu, y_data_gpu, "Speedup for varying maximum skip", "Skips allowed", "CPU/GPU", "SkipCompareSpeedup", True)
f = open('benchmarks_v3/skip_merged')
j = json.loads("".join(f.readlines()))

x_data = j['skip_sizes']
y_data_cpu = [sum([int(y) for y in x['sample']])/len(x['sample']) for x in j['cpu']]
y_data_gpu = [sum([int(y) for y in x['sample']])/len(x['sample']) for x in j['gpu']]

make_graph(x_data, y_data_cpu, y_data_gpu, "Execution time for maximum skip", "Skips allowed", "Time in ms", "SkipSampleCompare")

plot_speedup(x_data, y_data_cpu, y_data_gpu, "Speedup for varying maximum skip", "Skips allowed", "CPU/GPU", "SkipSampleCompareSpeedup", True)


f = open('benchmarks_v3/kmer_merged')
j = json.loads("".join(f.readlines()))

x_data = j['kmer_sizes']
y_data_cpu = [sum([int(y) for y in x['fw']])/len(x['fw']) for x in j['cpu']]
y_data_gpu = [sum([int(y) for y in x['fw']])/len(x['fw']) for x in j['gpu']]

make_graph(x_data[:5], y_data_cpu[:5], y_data_gpu[:5], "Execution time for kmer sizes", "Kmer size", "Time in ms", "KmerCompareFW")

plot_speedup(x_data[:5], y_data_cpu[:5], y_data_gpu[:5], "Speedup for varying kmer sizes", "Kmer size", "CPU/GPU", "KmerCompareFWSpeedup", True)

x_data = j['kmer_sizes']
y_data_cpu = [sum([int(y) for y in x['sample']])/len(x['sample']) for x in j['cpu']]
y_data_gpu = [sum([int(y) for y in x['sample']])/len(x['sample']) for x in j['gpu']]

print(y_data_cpu)
print(y_data_gpu)

make_graph(x_data[:5], y_data_cpu[:5], y_data_gpu[:5], "Execution time for kmer sizes", "Kmer size", "Time in ms", "KmerCompareSampling")

plot_speedup(x_data[:5], y_data_cpu[:5], y_data_gpu[:5], "Speedup for varying kmer sizes", "Kmer size", "CPU/GPU", "KmerCompareSamplingSpeedup", True)


f = open('benchmarks_v3/calibration_14_05_2017 14:33:21')
j = json.loads("".join(f.readlines()))

x_data = j['data_sizes']
y_data_cpu = [int(x['total']) for x in j['cpu']]
y_data_gpu = [int(x['total']) for x in j['gpu']]

make_graph(x_data, y_data_cpu, y_data_gpu, "Comparison of raw power", "Data size", "Time in ms", "CalibrationCompare")

plot_speedup(x_data, y_data_gpu, y_data_cpu, "CPU vs GPU raw performance", "Data size", "GPU/CPU ratio", "CalibrationCompareSpeedup", True)

"""

f = open('benchmarks_v3/decode_comparison_16_05_2017 16:52:49')
j = json.loads("".join(f.readlines()))

x_data = j['input_sizes']
y_data_cpu = [int(x['decode']) for x in j['cpu']]
y_data_gpu = [int(x['decode']) for x in j['gpu']]

vals = {}
indices = []
for i, val in enumerate(x_data):
	if val in vals:
		continue
	vals[val] = True
	indices.append(i)
x_data = [x_data[i] for i in indices]
y_data_cpu = [y_data_cpu[i] for i in indices]
y_data_gpu = [max(y_data_gpu[i],1) for i in indices]

combined = [(x,c,g) for x,c,g in zip(x_data, y_data_cpu, y_data_gpu)]
combined.sort()

x_data = []
y_data_cpu = []
y_data_gpu = []
for x,c,g in combined:
	x_data.append(x)
	y_data_cpu.append(c)
	y_data_gpu.append(g)

print(x_data)
print(y_data_cpu)
print(y_data_gpu)

make_graph(x_data, y_data_cpu, y_data_gpu, "Execution time for decoding phase", "Data size * number of samples", "Time in ms", "DecodeCompare")

plot_speedup(x_data, y_data_cpu, y_data_gpu, "Speedup of the decoding phase", "Data size * number of samples", "GPU/CPU ratio", "DecodeCompareSpeedup", True)