# coding: utf-8
import argparse, sys, os, time, re, codecs, random
import pandas as pd
import model

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def main(args):
	try:
		os.mkdir(args.model)
	except:
		pass
	assert args.filename is not None
	assert os.path.exists(args.filename)
	assert args.train_split is not None

	vpylm = model.vpylm()
	print stdout.BOLD + args.filename + "を読み込んでいます ..." + stdout.END
	assert vpylm.load_textfile(args.filename, args.train_split) == True
	print stdout.BOLD + args.filename + "を読み込みました." + stdout.END

	print "訓練データ数:	", vpylm.get_num_train_data() 
	print "テストデータ数:	", vpylm.get_num_test_data()
	print "語彙数:		", vpylm.get_num_types_of_words()
	print "総単語数:	", vpylm.get_num_words()

	vpylm.set_g0(1.0 / float(vpylm.get_num_types_of_words()))	# 規定分布を設定
	vpylm.compile()		# 初期化

	# グラフプロット用
	csv_likelihood = []
	csv_perplexity = []

	for epoch in xrange(1, args.epoch + 1):
		start = time.time()
		vpylm.perform_gibbs_sampling()
		vpylm.sample_hyperparameters()

		elapsed_time = time.time() - start
		sys.stdout.write("\rEpoch {} / {} - {:.3f} sec - {} depth - {} nodes - {} customers".format(epoch, args.epoch, elapsed_time, vpylm.get_vpylm_depth(), vpylm.get_num_nodes(), vpylm.get_num_customers()))		
		sys.stdout.flush()
		if epoch % 10 == 0:
			log_likelihood = vpylm.compute_log_Pdataset_test() 
			perplexity = vpylm.compute_perplexity_test()
			print "\nlog_likelihood:", int(log_likelihood), int(vpylm.compute_log_Pdataset_train())
			print "perplexity:", int(perplexity), int(vpylm.compute_perplexity_train())
			vpylm.save(args.model);
			counts = vpylm.count_tokens_of_each_depth()
			print counts
			# CSV出力
			csv_likelihood.append([epoch, log_likelihood])
			data = pd.DataFrame(csv_likelihood)
			data.columns = ["epoch", "log_likelihood"]
			data.to_csv("{}/likelihood.csv".format(args.model))
			csv_perplexity.append([epoch, perplexity])
			data = pd.DataFrame(csv_perplexity)
			data.columns = ["epoch", "perplexity"]
			data.to_csv("{}/perplexity.csv".format(args.model))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--filename", type=str, default=None, help="訓練用のテキストファイルのパス.")
	parser.add_argument("-e", "--epoch", type=int, default=1000, help="総epoch.")
	parser.add_argument("-m", "--model", type=str, default="out", help="保存フォルダ名.")
	parser.add_argument("-split", "--train-split", type=float, default=0.8, help="テキストデータうち何割を訓練データにするか.")
	main(parser.parse_args())