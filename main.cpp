#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <map>
#include <unordered_map> 
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <stdio.h>
#include <wchar.h>
#include <locale>
#include "node.h"
#include "vpylm.h"
#include "vocab.h"

using namespace std;

Vocab* load(string &filename, vector<wstring> &dataset){
	wifstream ifs(filename.c_str());
	wstring str;
	if (ifs.fail())
	{
		cout << "failed to load " << filename << endl;
		return NULL;
	}
	Vocab* vocab = new Vocab();
	while (getline(ifs, str))
	{
		dataset.push_back(str);
		for(int i = 0;i < str.length();i++){
			vocab->addCharacter(str[i]);
		}
	}
	cout << "loading " << dataset.size() << " lines." << endl;
	return vocab;
}

void show_progress(int step, int total, double &progress){
	progress = step / (double)(total - 1);
	int barWidth = 70;

	cout << "[";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) cout << "=";
		else if (i == pos) cout << ">";
		else cout << " ";
	}
	cout << "] " << int(progress * 100.0) << " %\r";
	cout.flush();

	progress += 0.16;
}


void vpylm_generate_sentence(Vocab* vocab, vector<wstring> &dataset, string model_dir){
	VPYLM* vpylm = new VPYLM();
	int num_chars = vocab->numCharacters();
	vpylm->_g0 = 1.0 / (double)num_chars;

	// 読み込み
	vpylm->load(model_dir);
	vocab->load(model_dir);

	int num_sample = 50;
	int max_length = 400;
	vector<id> sentence_char_ids;
	for(int s = 0;s < num_sample;s++){
		sentence_char_ids.clear();
		sentence_char_ids.push_back(vocab->bosId());
		for(int i = 0;i < max_length;i++){
			id word_id = vpylm->sampleNextWord(sentence_char_ids, vocab->eosId());
			sentence_char_ids.push_back(word_id);
			if(word_id == vocab->eosId()){
				break;
			}
		}
		wcout << vocab->characters2string(sentence_char_ids) << endl;
	}
}

void vpylm_visualize_order(Vocab* vocab, vector<wstring> &dataset, string model_dir){
	VPYLM* vpylm = new VPYLM();
	int num_chars = vocab->numCharacters();
	vpylm->_g0 = 1.0 / (double)num_chars;

	// 読み込み
	vpylm->load(model_dir);
	vocab->load(model_dir);

	int num_visualize = 50;

	vector<int> rand_perm;
	for(int i = 0;i < dataset.size();i++){
		rand_perm.push_back(i);
	}
	random_shuffle(rand_perm.begin(), rand_perm.end());

	vector<id> sentence_char_ids;
	vector<int> orders;
	for(int step = 0;step < num_visualize;step++){

		int data_index = rand_perm[step];

		wstring sentence = dataset[data_index];
		if(sentence.length() == 0){
			continue;
		}
		sentence_char_ids.clear();
		sentence_char_ids.push_back(vocab->bosId());
		for(int i = 0;i < sentence.length();i++){
			int id = vocab->char2id(sentence[i]);
			sentence_char_ids.push_back(id);
		}
		sentence_char_ids.push_back(vocab->eosId());

		orders.clear();
		for(int i = 0;i < sentence_char_ids.size();i++){
			int order = vpylm->sampleOrder(sentence_char_ids, i);
			orders.push_back(order);
		}
		wcout << vocab->characters2string(sentence_char_ids) << endl;
		for(int i = 0;i < orders.size();i++){
			cout << orders[i];
		}
		cout << endl << endl;
	}
}

void train_vpylm(Vocab* vocab, vector<wstring> &dataset, string model_dir){
	// 文脈木
	VPYLM* vpylm = new VPYLM();
	int num_chars = vocab->numCharacters();
	vpylm->_g0 = 1.0 / (double)num_chars;
	cout << "g0: " << vpylm->_g0 << endl;

	vector<id> sentence_char_ids;

	int max_epoch = 500;
	int train_per_epoch = dataset.size();

	vector<int> rand_perm;
	for(int i = 0;i < dataset.size();i++){
		rand_perm.push_back(i);
	}
	random_shuffle(rand_perm.begin(), rand_perm.end());

	unordered_map<int, vector<int> > prev_orders;

	// 読み込み
	vpylm->load(model_dir);
	vocab->load(model_dir);
	string filename = model_dir + "trainer";
	std::ifstream ifs(filename);
	if(ifs.good()){
		boost::archive::binary_iarchive iarchive(ifs);
		iarchive >> prev_orders;
	}

	printf("training in progress...\n");
	for(int epoch = 1;epoch <= max_epoch;epoch++){
		auto start_time = chrono::system_clock::now();
		double progress = 0.0;
		random_shuffle(rand_perm.begin(), rand_perm.end());

		for(int step = 0;step < train_per_epoch;step++){

			show_progress(step, train_per_epoch, progress);

			int data_index = rand_perm[step];

			wstring sentence = dataset[data_index];
			if(sentence.length() == 0){
				continue;
			}
			sentence_char_ids.clear();
			sentence_char_ids.push_back(vocab->bosId());
			for(int i = 0;i < sentence.length();i++){
				int id = vocab->char2id(sentence[i]);
				sentence_char_ids.push_back(id);
			}
			sentence_char_ids.push_back(vocab->eosId());

			if(prev_orders.find(data_index) == prev_orders.end()){
				 vector<int> prev_order(sentence_char_ids.size(), -1);
				 prev_orders[data_index] = prev_order;
			}else{
				vector<int> &prev_order = prev_orders[data_index];
				if(prev_order.size() != sentence_char_ids.size()){
					printf("\x1b[41;97m");
					printf("WARNING");
					printf("\x1b[49;39m");
					printf(" prev_order missmatch.\n");
				}else{
					for(int c_t_i = 0;c_t_i < sentence_char_ids.size();c_t_i++){
						int n_t = prev_order[c_t_i];
						bool success = vpylm->remove(sentence_char_ids, c_t_i, n_t);
						if(success == false){
							printf("\x1b[41;97m");
							printf("WARNING");
							printf("\x1b[49;39m");
							printf(" Failed to remove a customer from VPYLM.\n");
						}
					}				
				}
			}

			vector<int> &prev_order = prev_orders[data_index];
			if(prev_order.size() != sentence_char_ids.size()){
				printf("\x1b[41;97m");
				printf("WARNING");
				printf("\x1b[49;39m");
				printf(" prev_order missmatch.\n");
			}else{
				for(int c_t_i = 0;c_t_i < sentence_char_ids.size();c_t_i++){
					int n_t = vpylm->sampleOrder(sentence_char_ids, c_t_i);
					vpylm->add(sentence_char_ids, c_t_i, n_t);
					prev_order[c_t_i] = n_t;
				}
			}
		}

		vpylm->sampleHyperParams();

		auto end_time = chrono::system_clock::now();
		auto dur = end_time - start_time;
		auto msec = chrono::duration_cast<chrono::milliseconds>(dur).count();

		// パープレキシティ
		double vpylm_ppl = 0;
		for(int i = 0;i < train_per_epoch;i++){
			wstring sentence = dataset[rand_perm[i]];
			if(sentence.length() == 0){
				continue;
			}
			sentence_char_ids.clear();
			sentence_char_ids.push_back(vocab->bosId());
			for(int i = 0;i < sentence.length();i++){
				int id = vocab->char2id(sentence[i]);
				sentence_char_ids.push_back(id);
			}
			sentence_char_ids.push_back(vocab->eosId());
			double log_p = vpylm->log_Pw(sentence_char_ids) / (double)sentence_char_ids.size();
			vpylm_ppl += log_p;
		}
		vpylm_ppl = exp(-vpylm_ppl / (double)train_per_epoch);

		cout << endl << "[epoch " << epoch << "] " <<  (double)train_per_epoch / msec * 1000.0 << " sentences / sec; perplexity " << vpylm_ppl << endl;

		if(epoch % 100 == 0){
			vpylm->save(model_dir);
			vocab->save(model_dir);
		}
	}

	vpylm->save(model_dir);
	vocab->save(model_dir);
	std::ofstream ofs(filename);
	boost::archive::binary_oarchive oarchive(ofs);
	oarchive << prev_orders;

	// <!-- デバッグ用
	// 客を全て削除した時に客数が本当に0になるかを確認する場合
	for(int step = 0;step < train_per_epoch;step++){
		int data_index = rand_perm[step];

		wstring sentence = dataset[data_index];
		if(sentence.length() == 0){
			continue;
		}
		sentence_char_ids.clear();
		sentence_char_ids.push_back(vocab->bosId());
		for(int i = 0;i < sentence.length();i++){
			int id = vocab->char2id(sentence[i]);
			sentence_char_ids.push_back(id);
		}
		sentence_char_ids.push_back(vocab->eosId());

		if(prev_orders.find(data_index) != prev_orders.end()){
			vector<int> &prev_order = prev_orders[data_index];
			if(prev_order.size() != sentence_char_ids.size()){
				printf("\x1b[41;97m");
				printf("WARNING");
				printf("\x1b[49;39m");
				printf(" prev_order missmatch.\n");
			}else{
				for(int c_t_i = 0;c_t_i < sentence_char_ids.size();c_t_i++){
					int n_t = prev_order[c_t_i];
					bool success = vpylm->remove(sentence_char_ids, c_t_i, n_t);
					if(success == false){
						printf("\x1b[41;97m");
						printf("WARNING");
						printf("\x1b[49;39m");
						printf(" Failed to remove a customer from VPYLM.\n");
					}
				}
			}
		}
	}
	//  -->

	cout << vpylm->maxDepth() << endl;
	cout << vpylm->numChildNodes() << endl;
	cout << vpylm->numCustomers() << endl;
	cout << vpylm->_sum_stop_counts() << endl;
	cout << vpylm->_sum_pass_counts() << endl;
}

int main(int argc, char *argv[]){
	// 日本語周り
	setlocale(LC_CTYPE, "ja_JP.UTF-8");
	ios_base::sync_with_stdio(false);
	locale default_loc("ja_JP.UTF-8");
	locale::global(default_loc);
	locale ctype_default(locale::classic(), default_loc, locale::ctype); //※
	wcout.imbue(ctype_default);
	wcin.imbue(ctype_default);
	vector<wstring> dataset;

	// [arguments]
	// --textdir ****
	// 訓練データtrain.txtの入っているフォルダを指定

	string filename;
	string model_dir;
	cout << "num args = " << argc << endl;
	if(argc % 2 != 1){
		cout << "invalid command line arguments." << endl;
		return -1;
	}else{
		for (int i = 0; i < argc; i++) {
			cout << i << "-th args = " << argv[i] << endl; 
			if (string(argv[i]) == "--text_dir") {
				if(i + 1 >= argc){
					cout << "invalid command line arguments." << endl;
					return -1;
				}
				filename = string(argv[i + 1]) + string("/train.txt");
			}
			else if (string(argv[i]) == "--model_dir") {
				if(i + 1 >= argc){
					cout << "invalid command line arguments." << endl;
					return -1;
				}
				model_dir = string(argv[i + 1]) + string("/");
			}
		}
	}

	Vocab* vocab = load(filename, dataset);

	// train_vpylm(vocab, dataset, model_dir);
	vpylm_generate_sentence(vocab, dataset, model_dir);
	vpylm_visualize_order(vocab, dataset, model_dir);
	return 0;
}
