#include <boost/format.hpp>
#include "util.h"
#include "core/vpylm.h"

class Model{
public:
	string vpylm_filename = "model/vpylm.model";
	string trainer_filename = "model/vpylm.trainer";
	vector<vector<int>> prev_orders_for_data;
	VPYLM* vpylm;
	Model(double g0){
		c_printf("[*]%s\n", "VPYLMを初期化しています ...");
		vpylm = new VPYLM();
		vpylm->set_g0(g0);
		c_printf("[*]%s\n", (boost::format("G0 <- %lf") % g0).str().c_str());
		vpylm->load(vpylm_filename);
	}
	void init_trainer(vector<vector<id>> &dataset){
		prev_orders_for_data.clear();
		for(int data_index = 0;data_index < dataset.size();data_index++){
			vector<id> &token_ids = dataset[data_index];
			vector<int> prev_orders(token_ids.size(), -1);
			prev_orders_for_data.push_back(prev_orders);
		}
	}
	void load_trainer(){
		std::ifstream ifs(trainer_filename);
		if(ifs.good()){
			boost::archive::binary_iarchive iarchive(ifs);
			iarchive >> prev_orders_for_data;
		}
	}
	void save_model(){
		vpylm->save(vpylm_filename);
	}
	void save_trainer(){
		std::ofstream ofs(trainer_filename);
		boost::archive::binary_oarchive oarchive(ofs);
		oarchive << prev_orders_for_data;
	}
	void train(Vocab* vocab, vector<vector<id>> &dataset){
		init_trainer(dataset);
		load_trainer();
		vector<int> rand_indices;
		for(int data_index = 0;data_index < dataset.size();data_index++){
			rand_indices.push_back(data_index);
		}
		int max_epoch = 100;
		int num_data = dataset.size();

		for(int epoch = 1;epoch <= max_epoch;epoch++){
			auto start_time = chrono::system_clock::now();
			random_shuffle(rand_indices.begin(), rand_indices.end());

			for(int step = 0;step < num_data;step++){
				show_progress(step, num_data);
				int data_index = rand_indices[step];
				vector<id> &token_ids = dataset[data_index];
				vector<int> &prev_orders = prev_orders_for_data[data_index];

				for(int token_t_index = 0;token_t_index < token_ids.size();token_t_index++){
					if(prev_orders[token_t_index] != -1){
						vpylm->remove_customer_at_timestep(token_ids, token_t_index, prev_orders[token_t_index]);
					}
					int new_order = vpylm->sample_order_at_timestep(token_ids, token_t_index);
					vpylm->add_customer_at_timestep(token_ids, token_t_index, new_order);
					prev_orders[token_t_index] = new_order;
				}
			}

			vpylm->sample_hyperparams();

			auto end_time = chrono::system_clock::now();
			auto duration = end_time - start_time;
			auto msec = chrono::duration_cast<chrono::milliseconds>(duration).count();

			// パープレキシティ
			double ppl = 0;
			for(int step = 0;step < num_data;step++){
				vector<id> &token_ids = dataset[step];
				double log_p = vpylm->log2_Pw(token_ids) / token_ids.size();
				ppl += log_p;
			}
			ppl = exp(-ppl / num_data);
			printf("Epoch %d / %d - %.1f lps - %.3f ppl - %d depth - %d nodes - %d customers\n", epoch, max_epoch, (double)num_data / msec * 1000.0, ppl, vpylm->get_max_depth(), vpylm->get_num_nodes(), vpylm->get_num_customers());

			if(epoch % 100 == 0){
				save_model();
				save_trainer();
			}
		}
		save_model();
		save_trainer();

		// <!-- デバッグ用
		//客を全て削除した時に客数が本当に0になるかを確認する場合
		// for(int step = 0;step < num_data;step++){
		// 	int data_index = rand_indices[step];
		// 	vector<id> token_ids = dataset[data_index];
		// 	for(int token_t_index = ngram - 1;token_t_index < token_ids.size();token_t_index++){
		// 		vpylm->remove_customer_at_timestep(token_ids, token_t_index);
		// 	}
		// }
		//  -->

		cout << vpylm->get_max_depth() << endl;
		cout << vpylm->get_num_nodes() << endl;
		cout << vpylm->get_num_customers() << endl;
		cout << vpylm->get_sum_stop_counts() << endl;
		cout << vpylm->get_sum_pass_counts() << endl;
	}
	void enumerate_phrases_at_depth(Vocab* vocab, int depth, wstring spacer){
		c_printf("[*]%s\n", (boost::format("深さ%dの句を表示します ...") % depth).str().c_str());
		vector<vector<id>> phrases;
		vpylm->enumerate_phrases_at_depth(depth, phrases);
		for(int i = 0;i < phrases.size();i++){
			vector<id> &phrase = phrases[i];
			for(int j = 0;j < phrase.size();j++){
				wstring str = vocab->token_id_to_string(phrase[j]);
				wcout << str << spacer;
			}
			cout << endl;
		}
	}
	void generate_words(Vocab* vocab, wstring spacer){
		c_printf("[*]%s\n", "文章を生成しています ...");
		int num_sample = 50;
		int max_length = 400;
		id bos_id = vocab->string_to_token_id(L"<bos>");
		id eos_id = vocab->string_to_token_id(L"<eos>");
		vector<id> token_ids;
		for(int s = 0;s < num_sample;s++){
			token_ids.clear();
			token_ids.push_back(bos_id);
			for(int i = 0;i < max_length;i++){
				id token_id = vpylm->sample_next_token(token_ids, eos_id);
				token_ids.push_back(token_id);
				if(token_id == eos_id){
					break;
				}
			}
			for(auto token_id: token_ids){
				if(token_id == bos_id){
					continue;
				}
				if(token_id == eos_id){
					continue;
				}
				wstring word = vocab->token_id_to_string(token_id);
				wcout << word << spacer;
			}
			cout << endl;
		}
	}
};

int main(int argc, char *argv[]){
	// 日本語周り
	setlocale(LC_CTYPE, "ja_JP.UTF-8");
	ios_base::sync_with_stdio(false);
	locale default_loc("ja_JP.UTF-8");
	locale::global(default_loc);
	locale ctype_default(locale::classic(), default_loc, locale::ctype); //※
	wcout.imbue(ctype_default);
	wcin.imbue(ctype_default);

	string text_filename;
	cout << "num args = " << argc << endl;
	if(argc % 2 != 1){
		c_printf("[r]%s [*]%s\n", "エラー:", "テキストファイルを指定してください. -t example.txt");
		exit(1);
	}else{
		for(int i = 0; i < argc; i++){
			cout << i << "-th args = " << argv[i] << endl; 
			if (string(argv[i]) == "-t" || string(argv[i]) == "--text"){
				if(i + 1 >= argc){
					c_printf("[r]%s [*]%s %s\n", "エラー:", "不正なコマンドライン引数です.", string(argv[i]).c_str());
					exit(1);
				}
				text_filename = string(argv[i + 1]);
			}
		}
	}
	vector<vector<id>> dataset;
	Vocab* vocab;
	load_words_in_textfile(text_filename, dataset, vocab, 1);

	string vocab_filename = "model/vpylm.vocab";
	vocab->load(vocab_filename);
	vocab->save(vocab_filename);

	int num_chars = vocab->num_tokens();
	double g0 = (1.0 / num_chars);
	Model* vpylm = new Model(g0);
	// vpylm->train(vocab, dataset);
	vpylm->generate_words(vocab, L" ");
	vpylm->enumerate_phrases_at_depth(vocab, 6, L" ");
	return 0;
}
