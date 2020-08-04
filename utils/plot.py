###
#
# Plotting
# Support functions
#
####

import csv
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .spatial import explained_variance


def word_pair_dists_mean_var(dists_file_dict, model_filter=''):
    dists_dict = {dists_name: {} for dists_name, _ in dists_file_dict.items()}
    
    for dists_name, dists_file in dists_file_dict.items():
        with open(dists_file, 'r') as f:
            data = csv.reader(f)
            header = next(data)
            cols = {w:i for i,w in enumerate(header)}
            
            # Filtered header
            if not model_filter:
                header = header[2:]
            else:
                header = [i for i in header if re.search(model_filter, i)]
            
            print('cols', cols)
            print('header', header)
            
            # Column 0 : mean
            # Column 1 : var
            # Column 2 : std
            mean_var_dict = {dist_name: [0., 0., 0.] for dist_name in header}
            
            num_data = 0
            
            for row in data:
                for col in header:
                    mean_var_dict[col][0] += float(row[cols[col]])
                num_data += 1
                
            for col in header:
                mean_var_dict[col][0] /= num_data
            
            f.seek(0)
            next(data)
            
            for row in data:
                for col in header:
                    mean_var_dict[col][1] += (float(row[cols[col]]) - mean_var_dict[col][0])**2

            for col in header:
                mean_var_dict[col][1] /= num_data
                mean_var_dict[col][2] = np.sqrt(mean_var_dict[col][1])
                
            # print('num_data', num_data)
            # print('mean_var_dict', mean_var_dict)
        dists_dict[dists_name] = mean_var_dict
        
    return dists_dict


def plot_mean_vars(mean_var_dict_dict, save_fig_file, model_filter='', plot=False):
    
    fig, ax = plt.subplots()
    
    markers = ['o', 's', 'X', '^']
    num_datasets = len(mean_var_dict_dict)
    sep = 0.2
    pos = -(num_datasets - 1) * sep / 2
    i = 0
    
    for name, mean_var_dict in mean_var_dict_dict.items():
        labels = []
        means = []
        variances = []
        stds = []
        
        for key, value in mean_var_dict.items():
            labels.append(re.sub(model_filter, '', key))
            means.append(value[0])
            variances.append(value[1])
            stds.append(value[2])
        
        xs = [(x+pos) for x in range(len(labels))]
        # xticks = []
        
        # if ratio_models:
        #     labels.reverse()
        #     means.reverse()
        #     variances.reverse()
        #     stds.reverse()
        
        ax.errorbar(xs, means, stds, linestyle='None', marker=markers[i], label=name)
        
        i += 1
        pos += sep
    
    x_ticks = [x for x in range(len(labels))]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, rotation='vertical')
    # ax.set_yticks(ax.get_yticks())
    # ax.set_yticklabels(ax.get_yticklabels(), rotation='vertical')
    ax.tick_params(axis='y', rotation=90)
    ax.legend()
    ax.set_ylabel('Distance')
    plt.subplots_adjust(bottom=0.22)
    plt.subplots_adjust(top=0.97)
    plt.subplots_adjust(left=0.08)
    plt.subplots_adjust(right=0.98)
    
    if not save_fig_file is None:
        plt.savefig(save_fig_file)
    
    if plot: plt.show()


def word_zipf_distribution(vocab_file, a=1.5, verbose=False, incl_stop=True, log_scale=True):
    """
    Plot the (sorted) word frequencies and
    fit the plot with a Zipf distribution
    
    Parameters
    ----------
    vocab_file : str
        path to the vocabulary file
    a : float, optional
        Zipf distribution parameter
        (default: 1.5)
    verbose : bool, optional
        whether to print out information
        about the document (default: False)
    incl_stop : bool, optional
        whether to include stop words in
        the plots (default: True)
    log_scale : bool, optional
        whether to use log-log scale
        (default: True)
    
    Notes
    -----
    The vocabulary file is assumed to be a
    CSV file with the following columns:
    - 0 : word
    - 1 : counts
    - 2 : frequency (counts/num. words)
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        
        i = 0
        total = 0.
        
        freqs = []
        freqs_no_stops = []
        labels = []
        labels_no_stops = []
        
        current = 1.
        
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        
        for row in data:
            i += 1
            freqs.append(float(row[2]))
            labels.append(row[0])
            
            if row[0] not in stopwords:
                freqs_no_stops.append(float(row[2]))
                labels_no_stops.append(row[0])
            else:
                if verbose:
                    print('STOP: ', row[0])
        
        if incl_stop:
            freqs = freqs_no_stops
            labels = labels_no_stops
        
        x = np.arange(len(freqs))
        
        plt.plot(x, freqs, 'b+-')
        plt.xticks(x, labels, rotation='vertical')
        
        # Zipf distribution
        x_zipf = x + 1
        y = x_zipf**(-a) / special.zetac(a)
        plt.plot(x, y/max(y), linewidth=2, color='r')
        
        if verbose:
            print(i, ' rows')
            print('freqs len: ', len(freqs))
            print('MAX Y ', max(y))
        
        plt.show()


def plot_dist_changes(sorted_dist_changes, title='', labels='', fig_file='plots/plot_dist_changes'):
    """
    Plots a list of (sorted) distance changes
    as a single line
    
    Requirements
    ------------
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    sorted_dist_changes : list[float] OR list[list[float]]
        list of (ideally sorted) distances to
        plot. If list is multidimensional this
        produces a plot for each column
    title : str
        title for the plot
    labels : str
        labels for each of the plots produced,
        if plotting multiple sets of distances
        this should be a list whose size matches
        the number of plots
    """
    
    num_plots = 1
    
    # If list is multidimensional, it contains
    # multiple distance comparisons
    if isinstance(sorted_dist_changes[0], list):
        num_plots = len(sorted_dist_changes)
    
    fig, ax = plt.subplots()
    
    for i in range(num_plots):
        if num_plots == 1:
            distances = sorted_dist_changes
            label = str(labels)
        else:
            distances = sorted_dist_changes[i]
            if len(labels) == num_plots:
                label = labels[i]
            ax.legend()
        
        words, dists = zip(*distances)
    
        x = list(range(len(dists)))
        width = 0.4
        
        ax.plot(x, dists, marker='.', color=np.random.rand(3,), label=label)
        
    ax.set_xlabel('Word pair (sorted)')
    ax.set_ylabel('Distance change')
    ax.set_title(title + ' distance changes')
    
    plt.show()
    

def plot_dists_dict_histogram(dists_dict, save_file=None, plot=False, title='Distance histogram', model_filter='', norm=False):
    """
    Plot histograms of distance ranges read
    from a collection of files contained in a
    dictionary of file names and filepaths
    
    Requirements
    ------------
    import numpy as np
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    dists_dict : {str: str}
        dictionary of histograms file paths where
        keys are model names and values are file
        paths
    save_file : str, optional
        path to save the plot to (default: None)
    plot : bool, optional
        if true renders plot to screen
        (default: False)
    title : str, optional
        title for the plot
    model_filter : str, optional
        string to match the name of the model
        against, if empty no filtering occurs
        (default: '')
    norm : bool, optional
        whether or not to normalise the values
        (default: False)
    """
    fig, ax = plt.subplots()
    
    for name, d_file in dists_dict.items():
        if re.search(model_filter, name):
            dist_hist = np.load(d_file)
            
            bins = dist_hist[0, 5:65, 0]
            bin_ticks = ["{0:.2f}".format(b) for b in bins]
            xs = np.arange(len(bin_ticks))
            # bins = np.linspace(bins_ar.min(), bins_ar.max(), bins_ar.shape[0])
            dists = dist_hist[0, 5:65, 1]
            
            if norm:
                dists = dists / dists.sum()
            
            ax.bar(xs, dists, alpha=0.5, label=name)    
    
    x_ticks = [x for x in xs if x % 10 == 1]
    ax.set_xticks(x_ticks)#xs)
    b_ticks = [bin_ticks[i] for i in x_ticks]
    print('bins', b_ticks)
    ax.set_xticklabels(b_ticks)
    
    ax.legend()
        
    ax.set_xlabel('Distance (bins)')
    if norm:
        ax.set_ylabel('Frequencies')
    else:
        ax.set_ylabel('Counts')
    
    ax.set_title(title)
    
    if not save_file is None:
        plt.savefig(save_file)
    
    if plot: plt.show()


def plot_dists_dict_histo_subplot(dists_dict, save_file=None, plot=False, title='Distance histogram', model_filter='', norm=False):
    """
    Plot histograms of distance ranges read
    from a collection of files contained in a
    dictionary of file names and filepaths.
    Plot in separate subplots
    
    Requirements
    ------------
    import numpy as np
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    dists_dict : {str: str}
        dictionary of histograms file paths where
        keys are model names and values are file
        paths
    save_file : str, optional
        path to save the plot to (default: None)
    plot : bool, optional
        if true renders plot to screen
        (default: False)
    title : str, optional
        title for the plot
    model_filter : str, optional
        string to match the name of the model
        against, if empty no filtering occurs
        (default: '')
    norm : bool, optional
        whether or not to normalise the values
        (default: False)
    """
    
    if model_filter != '':
        num_subplots = 0
        for name, d_file in dists_dict.items():
            if re.search(model_filter, name):
                num_subplots += 1
    else:
        num_subplots = len(dists_dict)
    
    rows = num_subplots
    cols = 1
    ax_num = 0
    colors = ['#383F51', '#DDDBF1', '#3C4F76', '#6DD3CE', '#AB9F9D', '#F25C54', '#323031', '#95190C']
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
    
    for name, d_file in dists_dict.items():
        if re.search(model_filter, name):
            dist_hist = np.load(d_file)
            
            bins = dist_hist[0, 5:65, 0]
            bin_ticks = ["{0:.2f}".format(b) for b in bins]
            xs = np.arange(len(bin_ticks))
            # bins = np.linspace(bins_ar.min(), bins_ar.max(), bins_ar.shape[0])
            dists = dist_hist[0, 5:65, 1]
            
            if norm:
                dists = dists / dists.sum()
            
            color_num = ax_num % len(colors)
            axs[ax_num].bar(xs, dists, alpha=0.8, label=name, color=colors[color_num])
            
            x_ticks = [x for x in xs if x % 10 == 1]
            axs[ax_num].set_xticks(x_ticks)#xs)
            b_ticks = [bin_ticks[i] for i in x_ticks]
            print('bins', b_ticks)
            axs[ax_num].set_xticklabels(b_ticks)
            # axs[ax_num].set_yticks([])
            axs[ax_num].spines['left'].set_color('none')
            axs[ax_num].spines['right'].set_color('none')
            axs[ax_num].spines['top'].set_color('none')
            axs[ax_num].spines['bottom'].set_color('none')
            axs[ax_num].legend(frameon=False)
                
            # axs[ax_num].set_xlabel('Distance (bins)')
            ax_num += 1
            
    if norm:
        fig.text(0.04, 0.5, 'Frequencies', va='center', rotation='vertical')
    else:
        fig.text(0.04, 0.5, 'Counts', va='center', rotation='vertical')
    
    fig.text(0.5, 0.04, 'Distance (bins)', ha='center', va='center')
    
    fig.suptitle(title)
    
    # axs[ax_num].set_title(title)
    
    if not save_file is None:
        plt.savefig(save_file)
    
    if plot: plt.show()


def plot_dists_histogram_csv(dist_file, save_file=None, plot=False, title='Distance histogram', model_filter='', dist_type='_cos'):
    """
    Plot histograms of distance ranges read
    from a file of word pair distances with
    (at least) one column whose header name
    contains the string defined in dist_type
    
    Requirements
    ------------
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    dist_file : str
        path to the file containing the
        distances to plot
    dist_type : str, optional
        identifier for the columns of interest
        (default: 'cos')
    """
    with open(dist_file, 'r') as f:
        data = csv.reader(f)
        header = next(data)
        # Get a dictionary of the columns
        # that correspond to a distance type
        cols = {w: i for i, w in enumerate(header) if re.search(dist_type, w) and re.search(model_filter, w)}
        
        print(cols)
        
        dists = []
        
        for row in data:
            temp  = []
            for i in cols.values():
                temp.append(float(row[i]))
                
            dists.append(temp)
        
    min_val = np.amin(dists)
    max_val = np.amax(dists)
    dists = list(map(list, zip(*dists)))
    
    print('Min val: ', min_val)
    print('Max val: ', max_val)
    print('Dists shape: %dx%d' % (len(dists), len(dists[0])))
    
    fig, ax = plt.subplots()
    
    bins = np.linspace(min_val, max_val, 100)
    emb_names = list(cols.keys())
    
    for i, dist in enumerate(dists):
        ax.hist(dist, bins, alpha=0.5, label=emb_names[i])
    
    ax.legend()
        
    ax.set_xlabel('Distance (bins)')
    ax.set_ylabel('Counts')
    ax.set_title(title)
        
    if not save_file is None:
        plt.savefig(save_file)
    
    if plot: plt.show()


def plot_histogram(data_dict, num_bins=100, save_file=None, plot=False, title='Histogram', model_filter='', norm=False, x_label='Bins'):
    """
    Plot histograms of values ranges read from
    a collection of files contained in a
    dictionary of file names and filepaths.
    Files should be 1-D numpy arrays
    
    Requirements
    ------------
    import numpy as np
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    data_dict : {str: str}
        dictionary of datafile paths where keys
        are model names and values are filepaths
    num_bins : int, optional
        number of bins to divide the data into
        (default: 100)
    save_file : str, optional
        path to save the plot to (default: None)
    plot : bool, optional
        if true renders plot to screen
        (default: False)
    title : str, optional
        title for the plot (default: 'Histogram')
    model_filter : str, optional
        string to match the name of the model
        against, if empty no filtering occurs
        (default: '')
    norm : bool, optional
        whether or not to normalise the values
        (default: False)
    x_label : str, optional
        text to display on the x-axis
        (default: 'Bins')
    """
    
    fig, ax = plt.subplots()
    
    for name, datafile in data_dict.items():
        if re.search(model_filter, name):
            values = np.load(datafile)
            
            ax.hist(values, bins=num_bins, density=norm, alpha=0.5, label=name)    
    
    ax.legend()
        
    ax.set_xlabel(x_label)
    if norm:
        ax.set_ylabel('Frequencies')
    else:
        ax.set_ylabel('Counts')
    ax.set_title(title)
    
    if not save_file is None:
        plt.savefig(save_file)
    
    if plot: plt.show()



def plot_correlation_data(data_file, score_type='similarity'):
    """
    TODO: Make this function more general
    
    Create a plot from a correlation
    data file in the format of
    correlation_data()
    
    Requirements
    ------------
    import csv
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    data_file : str
        path to the correlation data
        file with the following columns:
        - word_1
        - word_2
        - relatedness / similarity
        - cos_w2v
        - euc_w2v
        - cos_w2v_syns
        - euc_w2v_syns
        - cos_rand
        - euc_rand
    score_type : str, optional
        type of score being ploted. Options are:
        - similarity
        - relatedness
        - SimLex999
    """
    with open(data_file, 'r') as f:
        data = csv.reader(f)
        header = next(data)
        
        # Creates a dictionary of column names
        cols = {w: i for i, w in enumerate(header)}
        
        words = []
        scores = []
        w2v_cos = []
        w2v_euc = []
        w2v_syns_cos = []
        w2v_syns_euc = []
        rand_cos = []
        rand_euc = []
        
        for row in data:
            words.append(row[cols['word_1']] + ' - ' + row[cols['word_2']])
            scores.append(float(row[cols[score_type]]))
            w2v_cos.append(float(row[cols['cos_w2v']]))
            w2v_euc.append(float(row[cols['euc_w2v']]))
            w2v_syns_cos.append(float(row[cols['cos_w2v_syns']]))
            w2v_syns_euc.append(float(row[cols['euc_w2v_syns']]))
            rand_cos.append(float(row[cols['cos_rand']]))
            rand_euc.append(float(row[cols['euc_rand']]))
            
        fig_cos = plt.figure()
        fig_euc = plt.figure()
        ax_cos = fig_cos.add_subplot(111)
        ax_euc = fig_euc.add_subplot(111)
        
        ax_cos.scatter(scores, w2v_cos, c='r', marker='.', label='w2v')
        ax_cos.scatter(scores, w2v_syns_cos, c='b', marker='+', label='w2v syns')
        ax_cos.scatter(scores, rand_cos, c='g', marker='v', label='rand init')
        
        ax_cos.set_xlabel('Scores (' + score_type + ')')
        # plt.xticks(x)
        ax_cos.set_ylabel('Cosine distance')
        # plt.yscale('log')
        ax_cos.legend()
        ax_cos.set_title('Cosine distance - ' + score_type + ' scores')
        
        ax_euc.scatter(scores, w2v_euc, c='r', marker='.', label='w2v')
        ax_euc.scatter(scores, w2v_syns_euc, c='b', marker='+', label='w2v syns')
        ax_euc.scatter(scores, rand_euc, c='g', marker='v', label='rand init')
        
        ax_euc.set_xlabel('Scores (' + score_type + ')')
        # plt.xticks(x)
        ax_euc.set_ylabel('Euclidean distance')
        # plt.yscale('log')
        ax_euc.legend()
        ax_euc.set_title('Euclidean distance - ' + score_type + ' scores')
        
        plt.show()


def plot_correl_scores(values_dict, save_file, correl_data='SimLex', tick_prefix='Vocab-', baseline=''):
    """
    Plot correlation scores from the different
    models
    
    Requirements
    ------------
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    values : list[float]
        1-D or 2-D array of correlation
        scores. 2-D is ploted as Pearson's
        r and Spearman 
    """
    
    ticks = [tick_prefix + str(voc_num) for voc_num in values_dict.keys()]
    x = np.array(list(range(len(ticks))))
    print('Ticks: ', ticks)
    
    label_values = {}
    fig, ax = plt.subplots()
    
    baseline_vals = []
    
    if correl_data != 'ratios':
        for vocab_size, vocab_values in values_dict.items():
            print('Plotting vocabulary ', vocab_size)
            
            for correl_name, correl_dict in vocab_values.items():
                if correl_name == correl_data:
                    if not label_values:
                        label_values = {model_name: [] for model_name in correl_dict.keys() if model_name != baseline}
                    
                    for model_name, correl_value in correl_dict.items():
                        if model_name != baseline:
                            label_values[model_name].append(correl_value)
                        else:
                            baseline_vals.append(correl_value)
    else:
        for correl_type, correl_dict in values_dict.items():
            print('Plotting score for ', correl_type)
            if not label_values:
                label_values = {model_name: [] for model_name in correl_dict.keys()}
                
            for model_name, correl_value in correl_dict.items():
                label_values[model_name].append(correl_value)
    
    print('Label-values: ', label_values)
    
    pos_offset = -(.8/2)
    width = .8 / len(label_values)
    print('Width: ', width)
    
    # Fill pattern for the bar chart
    patterns = ['/', '', '', '.', '', 'x', '', '\\', '', 'o', '', '-', '', '+', 'O', '*']
    # colors = ['#F25C54', '#62BBC1', '#30332E', '#EC058E', '#E3B505', '#73FBD3', '#323031', '#95190C']
    # edge_colors = ['#FFFFFF', '#000000', '#FFFFFF', '#FFFFFF', '#000000', '#4C4C4C', '#FFFFFF', '#FFFFFF']
    colors = ['#383F51', '#DDDBF1', '#3C4F76', '#6DD3CE', '#AB9F9D', '#F25C54', '#323031', '#95190C', '#383F51', '#DDDBF1', '#3C4F76', '#6DD3CE', '#AB9F9D', '#F25C54', '#323031', '#95190C']
    edge_colors = ['#DDDBF1', '#383F51', '#FFFFFF', '#383F51', '#383F51', '#DDDBF1', '#FFFFFF', '#FFFFFF','#DDDBF1', '#383F51', '#FFFFFF', '#383F51', '#383F51', '#DDDBF1', '#FFFFFF', '#FFFFFF']
    
    for i, (label, values) in enumerate(label_values.items()):
        print('Position offset:', pos_offset)
        bars = ax.bar(x + pos_offset, values, width=width, label=label, color=colors[i], hatch=patterns[i], edgecolor=edge_colors[i])
        
        pos_offset += width
    
    if baseline != '':
        x_base = [x_i for x_i in x]
        x_base[0] -= .8 * .5
        x_base[-1] += .8 * .5
        print('X base:', x_base)
        ax.plot(x_base, baseline_vals, label=baseline, linestyle='--', color='r')
    
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.legend(loc='upper left', bbox_to_anchor=(0., 0.4))
    ax.set_title(correl_data + ' correlation scores')
    plt.savefig(save_file)


def plot_correlation_scores(values, labels, name=''):
    """
    Plot correlation scores from the different
    models
    
    Requirements
    ------------
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    values : list[float]
        1-D or 2-D array of correlation
        scores. 2-D is ploted as Pearson's
        r and Spearman 
    """
    
    if len(values[0]) == 2:
        width = 0.4
        pearson_vals, spearman_vals = zip(*values)
        x = np.array(list(range(len(labels))))
        print('pearson_vals: ', pearson_vals)
        print('spearman_vals: ', spearman_vals)
        fig, ax = plt.subplots()
        ax.bar(x - (width/2), pearson_vals, width=width, label='Pearson\'s r')
        ax.bar(x + (width/2), spearman_vals, width=width, label='Spearman\'s rho')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_title(name + ' correlation scores')
        plt.show()
    else:
        print('Dim values: ', len(values[0]))


################# SVD Explained Variance
    

def plot_exp_variance(S, label='', title='', log_scale=False, plot=False):
    '''
    Plot the explained variance as a function
    of the number of (sorted) singular values
    (i.e. adding all the singular values accounts
    for 100% of the variance)
    
    Requirements
    ------------
    import matplotlib.pyplot as plt
    import numpy as np
    from spatial import explained_variance
    
    Parameters
    ----------
    S : list[float]
        list of singular values
    label : str, optional
        label for the plotted line (default: '')
    title : str, optional
        plot title (default: '')
    log_scale : bool, optional
        if true, plots in logarithmic scale
        (default: False)
    plot : bool, optional
        if true renders plot to screen
        (default: False)
    '''
    sing_vals = range(1, len(S)+1)
    exp_vars = explained_variance(S, len(S))
    cum_exp_vars = [np.sum(exp_vars[:i]) for i in range(1, len(S)+1)]
    
    print('Exp vars: ', exp_vars)
    print('Cumulative exp vars: ', cum_exp_vars)
    
    plt.plot(sing_vals, cum_exp_vars, label=label, color=np.random.rand(3,))
    plt.legend()
    plt.title(title)
    if log_scale: plt.yscale('log')
    
    if plot: plt.show()


def plot_exp_variances(S_dict, save_file=None, model_filter='', title='Explained variances', log_scale=False, plot=False):
    '''
    Plot the explained variance as a function
    of the number of (sorted) singular values
    (i.e. adding all the singular values accounts
    for 100% of the variance)
    This applies to a dictionary containing the
    model names and filpaths to the S values
    
    Requirements
    ------------
    import matplotlib.pyplot as plt
    import numpy as np
    from spatial import explained_variance
    
    Parameters
    ----------
    S_dict : {str: str}
        dictionary containing model names as
        keys and filepaths to singular values
        as values
    save_file : str, optional
        file to save plot to (default: None)
    model_filter : str, optional
        string to match the name of the model
        against, if empty no filtering occurs
        (default: '')
    title : str, optional
        plot title (default: '')
    log_scale : bool, optional
        if true, plots in logarithmic scale
        (default: False)
    plot : bool, optional
        if true renders plot to screen
        (default: False)
    '''
    
    fig, ax = plt.subplots()
    
    for name, S_file in S_dict.items():
        if re.search(model_filter, name):
            S = np.load(S_file)
            
            sing_vals = range(1, len(S)+1)
            exp_vars = explained_variance(S, len(S))
            cum_exp_vars = [np.sum(exp_vars[:i]) for i in range(1, len(S)+1)]
            
            ax.plot(sing_vals, cum_exp_vars, label=name, color=np.random.rand(3,))
    
    ax.legend()
        
    ax.set_xlabel('Singular values')
    if log_scale:
        ax.set_ylabel('Explained variance (log)')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('Explained variance')
    ax.set_title(title)
    
    if not save_file is None:
        plt.savefig(save_file)
    
    if plot: plt.show()


def plot_norms_histogram(norms_file_list, norms_dir, label='', title='Norms histogram', plot=True):
    """
    Plot histograms of embedding norm ranges
    read from a file of embedding norms
    
    Requirements
    ------------
    import numpy as np
    import matplotlib.pyplot as plt
    
    Parameters
    ----------
    norms_file_list : list[list[str]]
        list of norm name-directory to plot, 
        the format of this list is expected
        as follows:
        - [['norm 1 name', 'norm 1 file'], ...]
    norms_dir : str
        directory containing the files specified
        in norms_file_list
    label : str, optional
        label for the plotted line (default: '')
    title : str, optional
        plot title (default: 'Norms histogram')
    plot : bool, optional
        if true renders plot to screen
    """
    norms_list = []
    labels = []
    
    for name, norms_file in norms_file_list:
        print('Processing ', name, norms_file)
        norms_tmp = np.load(norms_dir + norms_file)
        print('Norms tmp shape: ', norms_tmp.shape)
        norms_list.append(norms_tmp)
        labels.append(name)
    
    norms_mtx = np.matrix(norms_list)
    
    print('Norms shape: ', norms_mtx.shape)
        
    min_val = np.amin(norms_mtx)
    max_val = np.amax(norms_mtx)
    
    print('Min val: ', min_val)
    print('Max val: ', max_val)
    
    fig, ax = plt.subplots()
    
    bins = np.linspace(min_val, max_val, 100)
    
    for i, norms in enumerate(norms_list):
        ax.hist(norms, bins, alpha=0.5, label=labels[i])
    
    ax.legend()
        
    ax.set_xlabel('Norm (bins)')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    
    if plot: plt.show()
