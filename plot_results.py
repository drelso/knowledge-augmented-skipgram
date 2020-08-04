###
#
# Plotting functions
#
###

from utils.plot import plot_correl_scores
from utils.model_paths import get_base_embeds, get_model_path, get_correl_dict
from utils.validation import correl_all_data_min_vocab, correl_coefficients_for_vocab


def correl_plots_vocabs():
    # embs_dict = get_model_path('embeds')
    # print(embs_dict)
    
    # base_embs = get_base_embeds()
    # print('\nBase embeds: ', base_embs)
    
    # vocab_num = 20
    # correl_dict = get_correl_dict(vocab_num)
    
    # print('\nCorrelation dictionary: ', correl_dict)
    
    # correl_all_data_min_vocab()
    vocab_sizes = [3, 7, 20]
    
    all_correl_results = {}
    for vocab_size in vocab_sizes:
        all_correl_results[vocab_size] = correl_coefficients_for_vocab(vocab_size, correl_metric='spearmanr')
    
    print('Full correl results:', all_correl_results)
    
    plots_dir = 'plots/correlation/'
    
    correl_data = 'WordSim-sim'#'SimLex'
    save_file = plots_dir + correl_data + '_full_correl.pdf'
    plot_correl_scores(all_correl_results, save_file, correl_data=correl_data)


def correl_plots_ratios():
    vocab_num = 7
    vocab_min = 20
    ratio_models = True
    
    embs_dict = get_model_path('embeds', ratio_models=ratio_models)
    correl_dict = get_correl_dict(vocab_num=7)
    print('Embs dict: ', embs_dict)
    print('Correl dict: ', correl_dict)
    # embs_dict = get_model_path('embeds')
    # print(embs_dict)
    
    base_embs = get_base_embeds()
    print('\nBase embeds: ', base_embs)
    
    # correl_dict = get_correl_dict(vocab_num)
    # print('\nCorrelation dictionary: ', correl_dict)
    
    # This calculates the distances for the word 
    # pairs in the correlation data and writes a
    # CSV file with the results
    correl_all_data_min_vocab(min_vocab_num=vocab_min, distance='cos', vocab_sizes=[vocab_num], ratio_models=ratio_models)
    
    all_correl_results = correl_coefficients_for_vocab(vocab_num, correl_metric='spearmanr', ratio_models=ratio_models)
    print('Full correl results:', all_correl_results)
    
    plots_dir = 'plots/correlation/'
    
    correl_data = 'ratios'#'SimLex'
    save_file = plots_dir + correl_data + '_full_correl.pdf'
    plot_correl_scores(all_correl_results, save_file, correl_data=correl_data, tick_prefix='')
    # plot_correl_scores(all_correl_results, save_file, correl_data='SimLex')


def correl_plots_vocabs(save_file):
    vocab_sizes = [3, 7, 20]
    vocab_min = 20
    ratio_models = False
    baseline = 'w2v'
    
    # This calculates the distances for the word 
    # pairs in the correlation data and writes a
    # CSV file with the results
    correl_all_data_min_vocab(min_vocab_num=vocab_min, distance='cos', vocab_sizes=vocab_sizes, ratio_models=ratio_models)
    
    all_correl_results = {}
    for vocab_size in vocab_sizes:
        all_correl_results[vocab_size] = correl_coefficients_for_vocab(vocab_size, correl_metric='spearmanr')
    print('Full correl results:', all_correl_results)
    
    # plot_correl_scores(all_correl_results, save_file, correl_data=correl_data, tick_prefix='')
    print('Plotting and saving plot to', save_file)
    plot_correl_scores(all_correl_results, save_file, correl_data='SimLex', baseline=baseline)


if __name__ == '__main__':
    # correl_plots_vocabs()
    
    # correl_plots_ratios()
    
    plots_dir = 'plots/correlation/'
    file_name = 'SimLex_full_correl_base'
    save_file = plots_dir + file_name + '.pdf'
    correl_plots_vocabs(save_file)
    