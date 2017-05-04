#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <unordered_map> 
#include <unordered_set>
#include <vector>
#include <cassert>
#include <fstream>
#include "sampler.h"
#include "common.h"
#include "node.h"

class VPYLM{
public:
	Node* _root;				// 文脈木のルートノード
	int _depth;					// 最大の深さ
	double _g0;					// ゼログラム確率

	// 深さmのノードに関するパラメータ
	vector<double> _d_m;		// Pitman-Yor過程のディスカウント係数
	vector<double> _theta_m;	// Pitman-Yor過程の集中度

	// "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C参照
	// http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
	vector<double> _a_m;		// ベータ分布のパラメータ	dの推定用
	vector<double> _b_m;		// ベータ分布のパラメータ	dの推定用
	vector<double> _alpha_m;	// ガンマ分布のパラメータ	θの推定用
	vector<double> _beta_m;		// ガンマ分布のパラメータ	θの推定用
	double _beta_stop;		// 停止確率q_iのベータ分布の初期パラメータ
	double _beta_pass;		// 停止確率q_iのベータ分布の初期パラメータ
	// 計算高速化用
	int _max_depth;
	double* _sampling_table;

	VPYLM(){
		_root = new Node(0);
		_root->_depth = 0;	// ルートは深さ0
		// http://www.ism.ac.jp/~daichi/paper/ipsj07vpylm.pdfによると初期値は(4, 1)
		// しかしVPYLMは初期値にあまり依存しないらしい
		_beta_stop = VPYLM_BETA_STOP;
		_beta_pass = VPYLM_BETA_PASS;
		_max_depth = 999;
		_sampling_table = new double[_max_depth];
	}
	~VPYLM(){
		_delete_node(_root);
		delete[] _sampling_table;
	}
	void _delete_node(Node* node){
		for(auto &elem: node->_children){
			Node* child = elem.second;
			_delete_node(child);
		}
		delete node;
	}
	bool add_customer_at_timestep(vector<id> &token_ids, int token_t_index, int depth_t){
		assert(0 <= depth_t && depth_t <= token_t_index);
		Node* node = find_node_by_tracing_back_context(token_ids, token_t_index, depth_t, true);
		assert(node != NULL);
		assert(node->_depth == depth_t);
		if(depth_t > 0){
			assert(node->_token_id == token_ids[token_t_index - depth_t]);
		}
		id token_t = token_ids[token_t_index];
		return node->add_customer(token_t, _g0, _d_m, _theta_m);
	}
	bool remove_customer_at_timestep(vector<id> &token_ids, int token_t_index, int depth_t){
		assert(0 <= depth_t && depth_t <= token_t_index);
		Node* node = find_node_by_tracing_back_context(token_ids, token_t_index, depth_t, true);
		assert(node != NULL);
		assert(node->_depth == depth_t);
		if(depth_t > 0){
			assert(node->_token_id == token_ids[token_t_index - depth_t]);
		}
		id token_t = token_ids[token_t_index];
		node->remove_customer(token_t);
		// 客が一人もいなくなったらノードを削除する
		if(node->need_to_remove_from_parent()){
			node->remove_from_parent();
		}
		return true;
	}
	int sample_depth_at_timestep(vector<id> &context_token_ids, int token_t_index){
		if(token_t_index == 0){
			return 0;
		}
		id token_t = context_token_ids[token_t_index];

		// 停止確率がこの値を下回れば打ち切り
		double eps = 1e-24;
		
		double sum = 0;
		double p_pass = 1;
		double pw = 0;
		int sampling_table_size = 0;
		Node* node = _root;
		for(int n = 0;n <= token_t_index;n++){
			if(node){
				pw = node->compute_Pw(token_t, _g0, _d_m, _theta_m);
				double p_stop = node->stop_probability(_beta_stop, _beta_pass, false) * p_pass;
				double p = pw * p_stop;
				p_pass *= node->pass_probability(_beta_stop, _beta_pass, false);
				_sampling_table[n] = p;
				sampling_table_size += 1;
				sum += p;
				if(p_stop < eps){
					break;
				}
				if(n < token_t_index){
					id context_token_id = context_token_ids[token_t_index - n - 1];
					node = node->find_child_node(context_token_id);
				}
			}else{
				double p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
				double p = pw * p_stop;
				_sampling_table[n] = p;
				sampling_table_size += 1;
				sum += p;
				p_pass *= _beta_pass / (_beta_stop + _beta_pass);
				if(p_stop < eps){
					break;
				}
			}
		}
		double normalizer = 1.0 / sum;
		double bernoulli = sampler::uniform(0, 1);
		double stack = 0;
		for(int n = 0;n < sampling_table_size;n++){
			stack += _sampling_table[n] * normalizer;
			if(bernoulli < stack){
				return n;
			}
		}
		return _sampling_table[sampling_table_size - 1];
	}
	double compute_Pw_given_h(id token_id, vector<id> &context_token_ids){
		assert(context_token_ids.size() > 0);
		Node* node = _root;

		// 停止確率がこの値を下回れば打ち切り
		double eps = 1e-24;

		double parent_pw = _g0;
		double p_pass = 1;
		double p_stop = 1;	// eps以上なら何でもいい
		double pw_h = 0;

		// 文脈の一時刻先を深さ0と考える
		// （token_idの位置が深さ0と考える）
		// 例）
		// context_token_ids: [1, 2, 3]
		// depth:              3  2  1  0
		int depth = 0;

		// 無限の深さまで考える
		// 実際のコンテキスト長を超えて確率を計算することもある
		while(p_stop > eps){
			if(node == NULL){	// 文脈ノードがない場合
				p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
				pw_h += parent_pw * p_stop;
				p_pass *= _beta_pass / (_beta_stop + _beta_pass);
			}else{
				assert(node->_depth == depth);
				double pw = node->compute_Pw_with_parent_Pw(token_id, parent_pw, _d_m, _theta_m);
				p_stop = node->stop_probability(_beta_stop, _beta_pass, false) * p_pass;
				p_pass *= node->pass_probability(_beta_stop, _beta_pass, false);
				pw_h += pw * p_stop;
				parent_pw = pw;
				if(depth < context_token_ids.size()){
					id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
					node = node->find_child_node(context_token_id);
				}else{
					node = NULL;	// 文脈を超えても計算する
				}
			}
			depth++;
		}
		assert(pw_h <= 1);
		return pw_h;
	}
	// old version
	double _compute_Pw_given_h(id token_id, vector<id> &context_token_ids){
		double eps = 1e-24;	// 停止確率がこの値を下回れば打ち切り
		double p = 0;		// 総和
		double p_stop = 1;	// eps以上なら何でもいい
		int n = 0;
		while(p_stop > eps){
			double pw = compute_Pw_given_hn(token_id, context_token_ids, n);
			p_stop = compute_Pn_given_h(n, context_token_ids);
			p += pw * p_stop;	// 深さで混合する
			n++;
		}
		return p;
	}
	double compute_Pw_given_hn(id token_id, vector<id> &context_token_ids, int n){
		Node* node = _root;
		for(int depth = 1;depth <= n;depth++){
			id context_token_id = context_token_ids[context_token_ids.size() - depth];
			Node* child = node->find_child_node(context_token_id);
			if(child == NULL){
				break;
			}
			node = child;
		}
		assert(node->_depth <= n);
		double p = node->compute_Pw(token_id, _g0, _d_m, _theta_m);
		return p;
	}
	double compute_Pn_given_h(int n, vector<id> &context_token_ids){
		Node* node = _root;
		double p_stop, p_pass;
		for(int depth = 0;depth <= n;depth++){
			if(node == NULL){
				p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
				p_pass *= _beta_pass / (_beta_stop + _beta_pass);
			}else{
				p_stop = node->stop_probability(_beta_stop, _beta_pass);
				p_pass = node->pass_probability(_beta_stop, _beta_pass);
				if(depth < context_token_ids.size()){
					id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
					node = node->find_child_node(context_token_id);
				}else{
					node = NULL;
				}
			}
		}
		return p_stop;
	}
	double compute_Pw(vector<id> &token_ids){
		if(token_ids.size() == 0){
			return 0;
		}
		double mult_pw = 1;
		vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + 1);
		for(int t = 1;t < token_ids.size();t++){
			id token_id = token_ids[t];
			double pw_h = compute_Pw_given_h(token_id, context_token_ids);
			assert(pw_h > 0);
			mult_pw *= pw_h;
			context_token_ids.push_back(token_ids[t]);
		}
		return mult_pw;
	}
	id sample_next_token(vector<id> &context_token_ids, unordered_set<id> &all_token_ids){
		vector<id> word_ids;
		vector<double> probs;
		double sum = 0;
		for(id token_id: all_token_ids){
			if(token_id == ID_BOS){
				continue;
			}
			double pw_h = compute_Pw_given_h(token_id, context_token_ids);
			if(pw_h > 0){
				word_ids.push_back(token_id);
				probs.push_back(pw_h);
				sum += pw_h;
			}
		}
		if(word_ids.size() == 0){
			return ID_EOS;
		}
		if(sum == 0){
			return ID_EOS;
		}
		double normalizer = 1.0 / sum;
		double bernoulli = sampler::uniform(0, 1);
		double stack = 0;
		for(int i = 0;i < word_ids.size();i++){
			stack += probs[i] * normalizer;
			if(stack >= bernoulli){
				return word_ids[i];
			}
		}
		return word_ids.back();;
	}

	// token列の位置tからorderだけ遡る
	// token_ids:        [0, 1, 2, 3, 4, 5]
	// token_t_index:4          ^     ^
	// order_t: 2               |<- <-|
	Node* find_node_by_tracing_back_context(vector<id> &token_ids, int token_t_index, int order_t, bool generate_node_if_needed = false, bool return_middle_node = false){
		if(token_t_index - order_t < 0){
			return NULL;
		}
		Node* node = _root;
		for(int depth = 1;depth <= order_t;depth++){
			id context_token_id = token_ids[token_t_index - depth];
			Node* child = node->find_child_node(context_token_id, generate_node_if_needed);
			if(child == NULL){
				if(return_middle_node){
					return node;
				}
				return NULL;
			}
			node = child;
		}
		return node;
	}
	double compute_log_Pw(vector<id> &token_ids){
		assert(token_ids.size() > 0);
		double sum_pw_h = 0;
		vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + 1);
		for(int t = 1;t < token_ids.size();t++){
			id token_id = token_ids[t];
			double pw_h = compute_Pw_given_h(token_id, context_token_ids);
			assert(pw_h > 0);
			sum_pw_h += log(pw_h);
			context_token_ids.push_back(token_id);
		}
		return sum_pw_h;
	}
	double compute_log2_Pw(vector<id> &token_ids){
		assert(token_ids.size() > 0);
		double sum_pw_h = 0;
		vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + 1);
		for(int t = 1;t < token_ids.size();t++){
			id token_id = token_ids[t];
			double pw_h = compute_Pw_given_h(token_id, context_token_ids);
			assert(pw_h > 0);
			sum_pw_h += log2(pw_h);
			context_token_ids.push_back(token_id);
		}
		return sum_pw_h;
	}
	void init_hyperparameters_at_depth_if_needed(int depth){
		if(depth >= _d_m.size()){
			while(_d_m.size() <= depth){
				_d_m.push_back(HPYLM_INITIAL_D);
			}
		}
		if(depth >= _theta_m.size()){
			while(_theta_m.size() <= depth){
				_theta_m.push_back(HPYLM_INITIAL_THETA);
			}
		}
		if(depth >= _a_m.size()){
			while(_a_m.size() <= depth){
				_a_m.push_back(HPYLM_INITIAL_A);
			}
		}
		if(depth >= _b_m.size()){
			while(_b_m.size() <= depth){
				_b_m.push_back(HPYLM_INITIAL_B);
			}
		}
		if(depth >= _alpha_m.size()){
			while(_alpha_m.size() <= depth){
				_alpha_m.push_back(HPYLM_INITIAL_ALPHA);
			}
		}
		if(depth >= _beta_m.size()){
			while(_beta_m.size() <= depth){
				_beta_m.push_back(HPYLM_INITIAL_BETA);
			}
		}
	}
	// "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C参照
	// http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
	void sum_auxiliary_variables_recursively(Node* node, vector<double> &sum_log_x_u_m, vector<double> &sum_y_ui_m, vector<double> &sum_1_y_ui_m, vector<double> &sum_1_z_uwkj_m){
		for(auto elem: node->_children){
			Node* child = elem.second;
			int depth = child->_depth;

			// if(depth > bottom){
			// 	bottom = depth;
			// }
			// init_hyperparameters_at_depth_if_needed(depth);

			double d = _d_m[depth];
			double theta = _theta_m[depth];
			sum_log_x_u_m[depth] += child->auxiliary_log_x_u(theta);	// log(x_u)
			sum_y_ui_m[depth] += child->auxiliary_y_ui(d, theta);		// y_ui
			sum_1_y_ui_m[depth] += child->auxiliary_1_y_ui(d, theta);	// 1 - y_ui
			sum_1_z_uwkj_m[depth] += child->auxiliary_1_z_uwkj(d);		// 1 - z_uwkj

			sum_auxiliary_variables_recursively(child, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m);
		}
	}
	// dとθの推定
	void sample_hyperparams(){
		int max_depth = get_depth();

		// 親ノードの深さが0であることに注意
		vector<double> sum_log_x_u_m(max_depth + 1, 0.0);
		vector<double> sum_y_ui_m(max_depth + 1, 0.0);
		vector<double> sum_1_y_ui_m(max_depth + 1, 0.0);
		vector<double> sum_1_z_uwkj_m(max_depth + 1, 0.0);

		// _root
		sum_log_x_u_m[0] = _root->auxiliary_log_x_u(_theta_m[0]);			// log(x_u)
		sum_y_ui_m[0] = _root->auxiliary_y_ui(_d_m[0], _theta_m[0]);		// y_ui
		sum_1_y_ui_m[0] = _root->auxiliary_1_y_ui(_d_m[0], _theta_m[0]);	// 1 - y_ui
		sum_1_z_uwkj_m[0] = _root->auxiliary_1_z_uwkj(_d_m[0]);				// 1 - z_uwkj

		// それ以外
		init_hyperparameters_at_depth_if_needed(max_depth);
		sum_auxiliary_variables_recursively(_root, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m);

		for(int u = 0;u <= max_depth;u++){
			_d_m[u] = sampler::beta(_a_m[u] + sum_1_y_ui_m[u], _b_m[u] + sum_1_z_uwkj_m[u]);
			_theta_m[u] = sampler::gamma(_alpha_m[u] + sum_y_ui_m[u], _beta_m[u] - sum_log_x_u_m[u]);
		}
	}
	int get_num_nodes(){
		return _root->get_num_nodes();
	}
	int get_num_customers(){
		return _root->get_num_customers();
	}
	int get_num_tables(){
		return _root->get_num_tables();
	}
	int get_sum_stop_counts(){
		return _root->sum_stop_counts();
	}
	int get_sum_pass_counts(){
		return _root->sum_pass_counts();
	}
	void update_max_depth(Node* node, int &max_depth){
		if(node->_depth > max_depth){
			max_depth = node->_depth;
		}
		for(const auto &elem: node->_children){
			Node* child = elem.second;
			update_max_depth(child, max_depth);
		}
	}
	int get_depth(){
		int max_depth = 0;
		update_max_depth(_root, max_depth);
		return max_depth;
	}
	void count_tokens_of_each_depth(unordered_map<int, int> &map){
		_root->count_tokens_of_each_depth(map);
	}
	void enumerate_phrases_at_depth(int depth, vector<vector<id>> &phrases){
		assert(depth <= _depth);
		// 指定の深さのノードを探索
		vector<Node*> nodes;
		_root->enumerate_nodes_at_depth(depth, nodes);
		for(auto &node: nodes){
			vector<id> phrase;
			while(node->_parent){
				phrase.push_back(node->_token_id);
				node = node->_parent;
			}
			phrases.push_back(phrase);
		}
	}
	template <class Archive>
	void serialize(Archive& archive, unsigned int version)
	{
		archive & _root;
		archive & _g0;
		archive & _beta_stop;
		archive & _beta_pass;
		archive & _d_m;
		archive & _theta_m;
		archive & _a_m;
		archive & _b_m;
		archive & _alpha_m;
		archive & _beta_m;
	}
	bool save(string filename = "hpylm.model"){
		std::ofstream ofs(filename);
		boost::archive::binary_oarchive oarchive(ofs);
		oarchive << *this;
		return true;
	}
	bool load(string filename = "hpylm.model"){
		std::ifstream ifs(filename);
		if(ifs.good() == false){
			return false;
		}
		boost::archive::binary_iarchive iarchive(ifs);
		iarchive >> *this;
		return true;
	}
};