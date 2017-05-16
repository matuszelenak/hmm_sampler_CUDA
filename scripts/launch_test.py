import subprocess

bashCommand = "./bachelor_thesis/src/hmm_sampler --model dataset/hmm6 --raw-input --scale dataset/_data_len_10000 --method GPU --sample 10 --maxskip 1 --version 2"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
output = output.decode("utf-8")
print(output)