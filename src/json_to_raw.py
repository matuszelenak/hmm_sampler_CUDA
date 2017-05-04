import json

s = "nanopore2_20161122_FNFAF01169_MN17633_sequencing_run_20161122_Human_Qiagen_1D_R9_4_65629_ch5_read16844_strand1.json"
f = open(s, 'r')

k = json.loads("".join(f.readlines()));
o = open(s + ".out", 'w+')
for mean in k['mean'][:100]:
	o.write(str(mean)+ "\n");
print(max(k['mean']), min(k['mean']))
f.close()
o.close()
