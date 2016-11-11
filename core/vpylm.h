#ifndef _vpylm_
#define _vpylm_
#include <vector>
#include <random>
#include <cmath>
#include <unordered_map> 
#include <fstream>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "c_printf.h"
#include "sampler.h"
#include "hpylm.h"
#include "node.h"

class VPYLM: public HPYLM{
private:
	friend class boost::serialization::access;
	template <class Archive>
	void serialize(Archive& archive, unsigned int version)
	{
		static_cast<void>(version); // No use
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
public:
	double _beta_stop;		// 停止確率q_iのベータ分布の初期パラメータ
	double _beta_pass;		// 停止確率q_iのベータ分布の初期パラメータ

	VPYLM(){
		_root = new Node(0);
		_root->_depth = 0;	// ルートは深さ0

		// http://www.ism.ac.jp/~daichi/paper/ipsj07vpylm.pdfによると初期値は(4, 1)
		// しかしVPYLMは初期値にあまり依存しないらしい
		_beta_stop = 4;
		_beta_pass = 1;
	}
	
	// order_tはw_tから見た深さ
	bool add_customer_at_timestep(vector<id> &token_ids, int token_t_index, int order_t){
		if(order_t > token_t_index){
			c_printf("[r]%s [*]%s\n", "エラー:", "客を追加できません. 不正な深さです.");
			return false;
		}
		Node* node = find_node_by_tracing_back_context(token_ids, token_t_index, order_t, true);
		if(node == NULL){
			c_printf("[r]%s [*]%s\n", "エラー:", "客を追加できません. ノードが見つかりません.");
			return false;
		}
		id token_t = token_ids[token_t_index];
		return node->add_customer(token_t, _g0, _d_m, _theta_m);
	}
	bool remove_customer_at_timestep(vector<id> &token_ids, int token_t_index, int order_t){
		Node* node = find_node_by_tracing_back_context(token_ids, token_t_index, order_t, true);
		if(node == NULL){
			c_printf("[r]%s [*]%s\n", "エラー:", "客を除去できません. ノードが見つかりません.");
			return false;
		}
		id token_t = token_ids[token_t_index];
		node->remove_customer(token_t);

		// 客が一人もいなくなったらノードを削除する
		if(node->need_to_remove_from_parent()){
			node->remove_from_parent();
		}
		return true;
	}
	int sample_order_at_timestep(vector<id> &context_token_ids, int token_t_index){
		if(token_t_index == 0){
			return 0;
		}
		id token_t = context_token_ids[token_t_index];
		vector<double> probs;
		double sum_p_stpp = 0;

		// この値を下回れば打ち切り
		double eps = 1e-6;
		
		double sum = 0;
		double p_pass = 0;
		double Pw = 0;
		Node* node = _root;
		for(int n = 0;n <= token_t_index;n++){
			if(node){
				Pw = node->Pw(token_t, _g0, _d_m, _theta_m);
				double p_stop = node->stop_probability(_beta_stop, _beta_pass);
				p_pass = node->pass_probability(_beta_stop, _beta_pass);
				double p = Pw * p_stop;
				probs.push_back(p);
				sum_p_stpp += p_stop;
				sum += probs[n];

				if(p < eps){
					break;
				}
				if(n < token_t_index){
					id context_token_id = context_token_ids[token_t_index - n - 1];
					node = node->find_child_node(context_token_id);
				}
			}else{
				double p_stop = p_pass * _beta_stop / (_beta_stop + _beta_pass);
				double p = Pw * p_stop;
				probs.push_back(p);
				sum_p_stpp += p_stop;
				sum += probs[n];
				p_pass *= _beta_pass / (_beta_stop + _beta_pass);
				if(p < eps){
					break;
				}
			}
		}
		double ratio = 1.0 / sum;
		uniform_real_distribution<double> rand(0, 1);
		double r = rand(Sampler::mt);
		sum = 0;
		for(int n = 0;n < probs.size();n++){
			sum += probs[n] * ratio;
			if(r < sum){
				return n;
			}
		}
		return probs.size() - 1;
	}
	double Pw_h(vector<id> &word_ids, vector<id> context_token_ids){
		double p = 1;
		for(int n = 0;n < word_ids.size();n++){
			p *= Pw_h(word_ids[n], context_token_ids);
			context_token_ids.push_back(word_ids[n]);
		}
		return p;
	}
	double Pw_h(id token_id, vector<id> &context_token_ids){
		Node* node = _root;
		int depth = 0;
		for(;depth < context_token_ids.size();depth++){
			id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
			if(node == NULL){
				break;
			}
			Node* child = node->find_child_node(context_token_id);
			if(child == NULL){
				break;
			}
			node = child;
		}
		double p = 0;
		for(int n = 0;n <= depth;n++){
			double a = Pw_hn(token_id, context_token_ids, n);
			double b = Pn_h(n, context_token_ids);
			p += a * b;
		}
		return p;
	}
	double Pw_hn(id token_id, vector<id> &context_token_ids, int n){
		if(n > context_token_ids.size()){
			c_printf("[r]%s [*]%s\n", "エラー:", "n > context_token_ids.size()");
			return 0;
		}
		Node* node = _root;
		int depth = 0;
		for(;depth < n;depth++){
			id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
			if(node == NULL){
				break;
			}
			Node* child = node->find_child_node(context_token_id);
			if(child == NULL){
				break;
			}
			node = child;
		}
		if(depth != n){
			c_printf("[r]%s [*]%s\n", "エラー:", "depth != n");
			return 0;
		}
		double p = node->Pw(token_id, _g0, _d_m, _theta_m);
		return p;
	}
	double Pn_h(int n, vector<id> &context_token_ids){
		if(n > context_token_ids.size()){
			c_printf("[r]%s [*]%s\n", "n > context_token_ids.size()");
			return 0;
		}
		Node* node = _root;
		int depth = 0;
		for(;depth < n;depth++){
			id context_token_id = context_token_ids[context_token_ids.size() - depth - 1];
			if(node == NULL){
				break;
			}
			Node* child = node->find_child_node(context_token_id);
			if(child == NULL){
				break;
			}
			node = child;
		}
		if(depth != n){
			c_printf("[r]%s [*]%s\n", "depth != n");
			return 0;
		}
		return node->stop_probability(_beta_stop, _beta_pass);
	}
	double Pw(vector<id> &word_ids){
		if(word_ids.size() == 0){
			return 0;
		}
		id w_0 = word_ids[0];
		double p0 = _root->Pw(w_0, _g0, _d_m, _theta_m) * _root->stop_probability(_beta_stop, _beta_pass);
		double p = p0;
		vector<id> context_token_ids(word_ids.begin(), word_ids.begin() + 1);
		for(int depth = 1;depth < word_ids.size();depth++){
			id word = word_ids[depth];
			double _p = Pw_h(word, context_token_ids);
			p *= _p;
			context_token_ids.push_back(word_ids[depth]);
		}
		return p;
	}
	double log_Pw(vector<id> &token_ids){
		double sum_Pw_h = 0;
		vector<id> context_token_ids(token_ids.begin(), token_ids.begin());
		for(int depth = 1;depth < token_ids.size();depth++){
			id token_id = token_ids[depth];
			double prob = Pw_h(token_id, context_token_ids);
			sum_Pw_h += log(prob + 1e-10);
			context_token_ids.push_back(token_id);
		}
		return sum_Pw_h;
	}
	double log2_Pw(vector<id> &token_ids){
		double sum_Pw_h = 0;
		vector<id> context_token_ids(token_ids.begin(), token_ids.begin());
		for(int depth = 1;depth < token_ids.size();depth++){
			id token_id = token_ids[depth];
			double prob = Pw_h(token_id, context_token_ids);
			sum_Pw_h += log2(prob + 1e-10);
			context_token_ids.push_back(token_id);
		}
		return sum_Pw_h;
	}
	id sample_next_token(vector<id> &context_token_ids, id eos_id){
		int token_t_index = context_token_ids.size() - 1;
		Node* node = _root;
		vector<double> probs;
		vector<Node*> nodes;
		double p = _root->stop_probability(_beta_stop, _beta_pass);
		probs.push_back(p);
		nodes.push_back(node);
		double sum = 0;

		for(int n = 0;n <= token_t_index;n++){
			if(node){
				id context_token_id = context_token_ids[token_t_index - n];
				node = node->find_child_node(context_token_id);
				if(node == NULL){
					break;
				}
				double p = node->stop_probability(_beta_stop, _beta_pass);
				probs.push_back(p);
				nodes.push_back(node);
				sum += p;
			}
		}
		if(sum == 0){
			return eos_id;
		}
		double ratio = 1.0 / sum;
		uniform_real_distribution<double> rand(0, 1);
		double r = rand(Sampler::mt);
		sum = 0;
		int depth = probs.size();
		for(int n = 0;n < probs.size();n++){
			sum += probs[n] * ratio;
			if(r < sum){
				depth = n;
			}
		}
		node = nodes[depth];

		vector<id> word_ids;
		probs.clear();
		sum = 0;
		for(auto elem: node->_arrangement){
			id token_id = elem.first;
			double p = Pw_h(token_id, context_token_ids);
			if(p > 0){
				word_ids.push_back(token_id);
				probs.push_back(p);
				sum += p;
			}
		}
		if(word_ids.size() == 0){
			return eos_id;
		}
		if(sum == 0){
			return eos_id;
		}
		ratio = 1.0 / sum;
		r = Sampler::uniform(0, 1);
		sum = 0;
		id sampled_word_id = word_ids.back();
		for(int i = 0;i < word_ids.size();i++){
			sum += probs[i] * ratio;
			if(sum > r){
				sampled_word_id = word_ids[i];
				break;
			}
		}
		return sampled_word_id;
	}
};

#endif