import sklearn.metrics as metrics
import brightside as bs
import numpy as np


def plot_action_cm(predictions, groundtruth, labels, show_title=True):
    fig, ax = bs.show_confusion_matrix(groundtruth, predictions, labels,
                                       figsize=(150, 150), annot=False,
                                       linewidths=0.005, linecolor='gray',
                                       square=True, show_xticks=True,
                                       cbar_kws={"shrink": 0.45})
    if show_title:
        acc = metrics.accuracy_score(groundtruth, predictions)
        ax.set_title('Action classification Acc. {:.2f}%'.format(acc * 100), y=1.02, fontsize=35)
    return fig, ax


def plot_verb_cm(predictions, groundtruth, labels, show_xticks=True, show_title=True, cmap='reds'):
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    if cmap == 'reds':
        cmap = list(reversed(sns.color_palette("Reds_r", 650).as_hex()))
        cmap[0] = '#EEEEEE'
        cmap = ListedColormap(cmap)
    else:
        cmap = None

    fig, ax = bs.show_confusion_matrix(predictions, groundtruth, labels,
                                       figsize=(8, 8), annot=False,
                                       linewidths=0.005, linecolor='gray',
                                       square=True, show_xticks=show_xticks,
                                       cbar_kws={"shrink": 0.75},
                                       cmap=cmap)
    if show_title:
        acc = metrics.accuracy_score(groundtruth, predictions)
        y = 1.15 if show_xticks else 1.0
        ax.set_title('Verb classification Acc. {:.2f}%'.format(acc * 100), y=y, fontsize=18)
    return fig, ax


def plot_noun_cm(predictions, groundtruth, labels, show_xticks=True, show_title=True, cmap='reds'):
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    if cmap == 'reds':
        cmap = list(reversed(sns.color_palette("Reds_r", 650).as_hex()))
        cmap[0] = '#EEEEEE'
        cmap = ListedColormap(cmap)
    else:
        cmap = None

    fig, ax = bs.show_confusion_matrix(predictions, groundtruth, labels,
                                       figsize=(14, 14), annot=False,
                                       linewidths=0.005, linecolor='gray',
                                       square=True, show_xticks=show_xticks,
                                       cbar_kws={"shrink": 0.75},
                                       cmap=cmap)
    if show_title:
        acc = metrics.accuracy_score(groundtruth, predictions)
        y = 1.15 if show_xticks else 1.0
        ax.set_title('Noun classification Acc. {:.2f}%'.format(acc * 100), y=y, fontsize=18)
    return fig, ax


def plot_baradel_verb_cm(predictions, groundtruth, labels, show_xticks=True, show_title=True):
    fig, ax = bs.show_confusion_matrix(predictions, groundtruth, labels,
                                       figsize=(19, 19), annot=False,
                                       linewidths=0.005, linecolor='gray',
                                       square=True, show_xticks=show_xticks,
                                       cbar_kws={"shrink": 0.75})

    if show_title:
        acc = metrics.accuracy_score(groundtruth, predictions)
        y = 1.07 if show_xticks else 1.0
        ax.set_title('Verb classification Acc. {:.2f}%'.format(acc * 100), y=y, fontsize=18)
    return fig, ax


def show_performance_measures(results, title='Performance'):
    bs.print_table(title, [('Top-1 accuracy:', '{:.2%}'.format(results.top1_acc)),
                           ('Top-5 accuracy:', '{:.2%}'.format(results.top5_acc)),
                           ('Avg. Class Precision:', '{:.2%}'.format(results.avg_precision)),
                           ('Avg. Class Recall:', '{:.2%}'.format(results.avg_recall))])


def show_action_performance_measures(results):
    show_performance_measures(results.verb, 'Verb performance')
    show_performance_measures(results.noun, 'Noun performance')
    show_performance_measures(results.action, 'Action performance')


def calculate_acc_diff(baseline_scores, scores, groundtruth):
    num_classes = baseline_scores.shape[1]

    baseline_predictions = baseline_scores.argmax(axis=1)
    baseline_cm = metrics.confusion_matrix(groundtruth, baseline_predictions, labels=np.arange(num_classes))

    predictions = scores.argmax(axis=1)
    cm = metrics.confusion_matrix(groundtruth, predictions, labels=np.arange(num_classes))

    baseline_tp = np.diag(baseline_cm)
    tp = np.diag(cm)

    total = np.sum(cm, axis=1)+1

    return 100 * (tp - baseline_tp) / total


def calculate_acc(scores, groundtruth):
    predictions = scores.argmax(axis=1)
    cm = metrics.confusion_matrix(groundtruth, predictions)
    tp = np.diag(cm)
    total = np.sum(cm, axis=1)
    return 100 * tp / total


def calculate_prec_diff(baseline_scores, scores, groundtruth):
    baseline_predictions = baseline_scores.argmax(axis=1)

    num_classes = baseline_scores.shape[1]
    baseline_precision = metrics.precision_score(groundtruth, baseline_predictions, labels=np.arange(num_classes), average=None)

    predictions = scores.argmax(axis=1)
    precision = metrics.precision_score(groundtruth, predictions, labels=np.arange(num_classes), average=None)
    return 100 * (precision-baseline_precision)


def _plot_diff(sorted_acc_diff, sorted_labels, ax, color='b', rotate_acc_annotation=False,
                   show_zero_acc_scores=True, labels_rotation_angle=90, ylabel="Accuracy difference %"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.barplot(x=np.arange(len(sorted_acc_diff)), y=sorted_acc_diff, ax=ax, color=color)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.yaxis.grid()
    ax.axhline(0, color="k", clip_on=False)
    ax.set_ylabel(ylabel, fontsize=16)

    for i, l in enumerate(sorted_labels):
        if sorted_acc_diff[i] >= 0:
            horizontalalignment = 'right'
            y = -1.8
        else:
            horizontalalignment = 'left'
            y = 1
        ax.text(0.2 + i, y, l, fontsize=14, rotation=labels_rotation_angle, rotation_mode='anchor',
                 horizontalalignment=horizontalalignment)

        if not show_zero_acc_scores and sorted_acc_diff[i] == 0:
            continue
        if rotate_acc_annotation:
            rotation = 90
            if sorted_acc_diff[i] > 0:
                y = 0.5
                format_num = '{:0.2f}'.format(sorted_acc_diff[i])
                verticalalignment = 'bottom'
            elif sorted_acc_diff[i] < 0:
                y = -.2
                format_num = '{:0.2f}'.format(sorted_acc_diff[i])
                verticalalignment = 'top'
            else:
                y = 0.7
                format_num = '{:0.1f}'.format(sorted_acc_diff[i])
                verticalalignment = 'bottom'
        else:
            rotation = 0
            verticalalignment = 'baseline'
            if sorted_acc_diff[i] > 0:
                y = 0.5
                format_num = '{:0.2f}'.format(sorted_acc_diff[i])
            elif sorted_acc_diff[i] < 0:
                y = -2.5
                format_num = '{:0.2f}'.format(sorted_acc_diff[i])
            else:
                y = 0.7
                format_num = '{:0.1f}'.format(sorted_acc_diff[i])
        ax.text(i, sorted_acc_diff[i] + y, format_num, fontsize=12, rotation=rotation, horizontalalignment='center',
                 verticalalignment=verticalalignment)


def plot_diff(sorted_acc_diff, sorted_labels, figsize=None, color="b", rotate_acc_annotation=False,
                  show_zero_acc_scores=True, split=False, hspace=None, labels_rotation_angle=90,
                  ylabel="Accuracy difference %"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    if not figsize:
        figsize = (18, 4)

    if split:
        fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
        sns.set_style("white", {'grid.linestyle': ':', 'palette': "colorblind", 'color_codes': True})
        if hspace:
            fig.subplots_adjust(hspace=hspace)
        pos_ind = np.nonzero(sorted_acc_diff > 0)[0]
        neg_ind = np.nonzero(sorted_acc_diff < 0)[0]
        _plot_diff(sorted_acc_diff[pos_ind], sorted_labels[pos_ind], ax[0], color, rotate_acc_annotation,
                       show_zero_acc_scores, labels_rotation_angle=labels_rotation_angle, ylabel=ylabel)
        _plot_diff(sorted_acc_diff[neg_ind], sorted_labels[neg_ind], ax[1], color, rotate_acc_annotation,
                       show_zero_acc_scores, labels_rotation_angle=labels_rotation_angle, ylabel=ylabel)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
        sns.set_style("white", {'grid.linestyle': ':', 'palette': "colorblind", 'color_codes': True})
        _plot_diff(sorted_acc_diff, sorted_labels, ax, color, rotate_acc_annotation, show_zero_acc_scores,
                   ylabel=ylabel)

    return fig, ax


def plot_class_acc(sorted_acc, sorted_labels, figsize=None, color="b", rotate_acc_annotation=False,
                   show_zero_acc_scores=True):
    import seaborn as sns
    import matplotlib.pyplot as plt

    if not figsize:
        figsize = (18, 4)

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    sns.set_style("white", {'grid.linestyle': ':', 'palette': "colorblind", 'color_codes': True})
    sns.barplot(x=np.arange(len(sorted_acc)), y=sorted_acc, ax=ax, color=color)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.yaxis.grid()
    ax.axhline(0, color="k", clip_on=False)
    ax.set_ylabel("Accuracy %", fontsize=16)

    for i, l in enumerate(sorted_labels):
        horizontalalignment = 'right'
        y = -0.1
        plt.text(0.2 + i, y, l, fontsize=14, rotation=90, rotation_mode='anchor',
                 horizontalalignment=horizontalalignment)

        if not show_zero_acc_scores and sorted_acc[i] == 0:
            continue
        if rotate_acc_annotation:
            rotation = 90
            if sorted_acc[i] > 0:
                y = 0.5
                format_num = '{:0.2f}'.format(sorted_acc[i])
                verticalalignment = 'bottom'
            else:
                y = 0.7
                format_num = '{:0.1f}'.format(sorted_acc[i])
                verticalalignment = 'bottom'
        else:
            rotation = 0
            verticalalignment = 'baseline'
            if sorted_acc[i] > 0:
                y = 0.5
                format_num = '{:0.2f}'.format(sorted_acc[i])
            else:
                y = 0.7
                format_num = '{:0.1f}'.format(sorted_acc[i])
        plt.text(i, sorted_acc[i] + y, format_num, fontsize=12, rotation=rotation, horizontalalignment='center',
                 verticalalignment=verticalalignment)
    return fig, ax


def plot_verb_acc_diff(baseline_scores, scores, groundtruth, labels, figsize=None, remove_zeros=False):
    acc_diff = calculate_acc_diff(baseline_scores, scores, groundtruth)
    sorted_ind = np.argsort(acc_diff)[::-1]
    sorted_acc_diff = acc_diff[sorted_ind]
    sorted_labels = [labels[i] for i in sorted_ind]
    if remove_zeros:
        indices=np.nonzero(sorted_acc_diff)[0]
        sorted_acc_diff = sorted_acc_diff[indices]
        sorted_labels = [sorted_labels[i] for i in indices]
    return plot_diff(sorted_acc_diff, sorted_labels, figsize=figsize)


def plot_noun_acc_diff(baseline_scores, scores, groundtruth, labels, figsize=None, remove_zeros=False):
    acc_diff = calculate_acc_diff(baseline_scores, scores, groundtruth)
    sorted_ind = np.argsort(acc_diff)[::-1]
    sorted_acc_diff = acc_diff[sorted_ind]
    sorted_labels = [labels[i] for i in sorted_ind]
    if remove_zeros:
        indices=np.nonzero(sorted_acc_diff)[0]
        sorted_acc_diff = sorted_acc_diff[indices]
        sorted_labels = [sorted_labels[i] for i in indices]
    return plot_diff(sorted_acc_diff, sorted_labels, figsize=figsize, color='r', rotate_acc_annotation=True,
                         show_zero_acc_scores=False)


def plot_action_acc_diff(baseline_scores, scores, groundtruth, labels, figsize=None):
    acc_diff = calculate_acc_diff(baseline_scores, scores, groundtruth)

    zero_indices = np.nonzero(acc_diff)
    acc_diff = acc_diff[zero_indices]

    labels = np.asarray(labels)
    labels = labels[zero_indices]

    sorted_ind = np.argsort(acc_diff)[::-1]
    sorted_acc_diff = acc_diff[sorted_ind]
    sorted_labels = labels[sorted_ind]
    return plot_diff(sorted_acc_diff, sorted_labels, figsize=figsize, color='g', rotate_acc_annotation=True,
                         show_zero_acc_scores=False, split=True, hspace=1.8, labels_rotation_angle=50)


def plot_action_prec_diff(baseline_scores, scores, groundtruth, labels, figsize=None):
    precision_diff = calculate_prec_diff(baseline_scores, scores, groundtruth)

    zero_indices = np.nonzero(precision_diff)
    precision_diff = precision_diff[zero_indices]

    labels_ = np.asarray(labels)
    labels_ = labels_[zero_indices]

    sorted_ind = np.argsort(precision_diff)[::-1]
    sorted_precision_diff = precision_diff[sorted_ind]
    sorted_labels = labels_[sorted_ind]

    return plot_diff(sorted_precision_diff, sorted_labels, figsize=figsize, color='g', rotate_acc_annotation=True,
                     show_zero_acc_scores=False, split=True, hspace=1.8, labels_rotation_angle=55,
                     ylabel='Precision diff%')


def plot_verb_acc(scores, groundtruth, labels, figsize=None):
    acc = calculate_acc(scores, groundtruth)
    sorted_ind = np.argsort(acc)[::-1]
    sorted_acc = acc[sorted_ind]
    sorted_labels = [labels[i] for i in sorted_ind]
    return plot_class_acc(sorted_acc, sorted_labels, figsize=figsize)


def plot_noun_acc(scores, groundtruth, labels, figsize=None):
    acc = calculate_acc(scores, groundtruth)
    sorted_ind = np.argsort(acc)[::-1]
    sorted_acc = acc[sorted_ind]
    sorted_labels = [labels[i] for i in sorted_ind]
    return plot_class_acc(sorted_acc, sorted_labels, figsize=figsize, color='r', rotate_acc_annotation=True,
                          show_zero_acc_scores=False)
