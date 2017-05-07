#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <boost/program_options.hpp>
#include <cassert>
#include <boost/log/trivial.hpp>

#include "HMM.h"
#include "./fast5/src/fast5.hpp"

#define BOOST_LOG_DYN_LINK 1

std::vector<double>load_fast5(std::string file_name, long long limit){
	std::vector<double>event_data;
	if (not fast5::File::is_valid_file(file_name))
    {
        BOOST_LOG_TRIVIAL(info) << "not a fast5 file [" << file_name << "]";
        return event_data;
    }
    {
        fast5::File f;
        try
        {
            f.open(file_name);
            assert(f.is_open());
            bool have_basecall_group = f.have_basecall_group();
            if (have_basecall_group)
            {
                auto bc_gr_list = f.get_basecall_group_list();
                for (unsigned st = 0; st < 3; ++st)
                {
                    auto gr_l = f.get_basecall_strand_group_list(st);
                    bool have_events = f.have_basecall_events(st);
                    if (have_events)
                    {
                        auto ev = f.get_basecall_events(st);
                        for (const auto& e : ev)
                        {
                        	event_data.push_back(e.mean);
                        }
                    }
                }
            }
        }
        catch (hdf5_tools::Exception& e)
        {
            BOOST_LOG_TRIVIAL(info) << "hdf5 error: " << e.what();
        }
    }
    assert(fast5::File::get_object_count() == 0);
    if (limit != -1){
    	event_data.resize(limit);
    }
    return event_data;
}

std::vector<double>loadEventData(std::string filename){
	BOOST_LOG_TRIVIAL(info) << "Loading event data from " << filename;
	std::ifstream event_file;
	event_file.open(filename);
	std::string line;
	std::vector<double>res;
	while(getline(event_file, line)) {
		std::size_t offset = 0;
		res.push_back(std::stod(&line[0], &offset));
	}
	BOOST_LOG_TRIVIAL(info) << "Loading done";
	return res;
}

void save_to_fasta(std::string filename, std::vector<std::string> seq_names, std::vector<std::vector<char> >seq_data){
	std::ofstream seq_file;
	seq_file.open(filename + ".fasta");
	for (int i = 0; i < seq_names.size(); i++){
		std::string name = seq_names[i] + ">\n";
		seq_file << name;
		for (int j = 0; j < seq_data[i].size(); j++){
			seq_file << seq_data[i][j];
		}
		seq_file << "\n";
	}
	seq_file.close();
	BOOST_LOG_TRIVIAL(info) << "Saved file " << filename + ".fasta";
}

int main(int argc, char const *argv[])
{
	namespace po = boost::program_options; 
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "Print usage")
		("input-file", po::value< std::vector<std::string> >()->required(), "Specify a list of input files")
		("model,m", po::value<std::string>(), "Name of HMM file to use as pore model")
		("sample,s", po::value<int>(), "If specified, number of samples to generate for each input file")
		("viterbi,v", "Run Viterbi for each input file")
		("fasta,f", "Save the results into FASTA file for each input file")
		("scale", "Adjust model scaling according to event data")
		("skip", po::value<double>(),"Set custom skip probability for model")
		("stay", po::value<double>(), "Set custom stay probability for model")
		("head,h",po::value<long long>(), "Set a limit on number of events processed")
		("method",po::value<std::string>(), "Method for calculation CPU|GPU")
	;
	po::positional_options_description p;
	p.add("input-file", -1);
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).
		options(desc).positional(p).run(), vm);
	po::notify(vm);    

	if (vm.count("help")) {
	    std::cout << desc << "\n";
	    return 1;
	}

	std::string model_name;
	if (vm.count("model")) {
		model_name = vm["model"].as<std::string>();
	} else {
	    BOOST_LOG_TRIVIAL(info) << "Model was not set. Defaulting to r73.c.p1.006.ont.model\n";
	    model_name = "r73.c.p1.006.ont.model";
	}

	HMM hmm;
	hmm.load_model_params(model_name);

	if (vm.count("skip")) hmm.set_skip_prob(vm["skip"].as<double>());

	if (vm.count("stay")) hmm.set_stay_prob(vm["stay"].as<double>());

	hmm.compute_transitions();

	long long limit = -1;
	if (vm.count("head")){
		limit = vm["head"].as<long long>();
	}

	std::vector<std::vector<double> >input_file_data;
	std::vector<std::string>input_file_names;
	if (vm.count("input-file")){
		input_file_names = vm["input-file"].as< std::vector<std::string> >();
		for (int f = 0; f < input_file_names.size(); f++){
			input_file_data.push_back(load_fast5(input_file_names[f], limit));
		}
	}
	std::string method;
	if (vm.count("method")){
		method = vm["method"].as< std::string >();
	}

	std::vector<std::vector<char> >viterbi_results;
	if (vm.count("viterbi")){
		for (int f = 0; f < input_file_data.size(); f++){
			if (vm.count("scale")){
				hmm.adjust_scaling(input_file_data[f]);
			}
			std::vector<int>v_path = hmm.compute_viterbi_path(input_file_data[f], method);
			for (int i = 0; i < v_path.size(); i++){
				printf("%d ", v_path[i]);
			}
			printf("\n");
			viterbi_results.push_back(hmm.translate_to_bases(v_path));
		}

		if (vm.count("fasta")){
			for (int i = 0; i < input_file_names.size(); i++){
				std::vector<std::string>seq_names = {input_file_names[i]};
				std::vector<std::vector<char> >seq_data = {viterbi_results[i]};
				save_to_fasta(input_file_names[i], seq_names, seq_data);
			}
		}
	}

	if (vm.count("sample")){
		int num_of_samples = vm["sample"].as<int>();
		for (int f = 0; f < input_file_data.size(); f++){
			if (vm.count("scale")){
				hmm.adjust_scaling(input_file_data[f]);
			}
			Matrix<int> samples = hmm.generate_samples(num_of_samples, input_file_data[f], method);
			Matrix<char>translated_samples(samples.size());
			std::transform(samples.begin(), samples.end(), translated_samples.begin(),
							[hmm](std::vector<int>x){return hmm.translate_to_bases(x);});

			if (vm.count("fasta")){
				std::vector<std::string>seq_names;
				for (int i = 0; i < samples.size(); i++){
					seq_names.push_back(input_file_names[f] + "_sample_" + std::to_string(i));
				}
				save_to_fasta(input_file_names[f], seq_names, translated_samples);
			}
		}
	}


}