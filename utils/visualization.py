import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch as t
from torch import Tensor
from jaxtyping import Float
import os
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import matplotlib as mpl
from utils.eval_circuit import  circuit_size, TPR, FPR
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import itertools
from matplotlib.lines import Line2D
from cycler import cycler
from utils.data_io import save_img, store_df
import pandas as pd
from matplotlib.ticker import MaxNLocator

title_font=16
fontsize=14
labelsize=12
cbar_fontsize=10
val_size=9

#----------------------------------------------------------------------------------------------------
# Heatmaps
#----------------------------------------------------------------------------------------------------

def heat_map_path_patching(
    scores: Float[Tensor, "layer head"], 
    title: str, 
    color_axis_title: str, 
    show: bool = True,
    save: bool = False,
    name: str = "",
    subfolder: str = "",
    senders: list = [], 
    print_scores=True, 
):
    """Heatmap (Layer x Heads) for Path Patching. Each score is the variation on the metric, if the associated head 
    is patched

    Args:
        scores ( Float[Tensor, "layer head"]): Influence scores for each head over a specific metric (usually avg_logit_diff)
        title (str): title of heatmap
        color_axis_title (str): different metrics, need different axis titles
        show (bool, optional): show. Defaults to True.
        save (bool, optional): save. Defaults to False.
        name (str, optional): Name under which file is saved. Defaults to "".
        subfolder (str, optional): subfolder where to save file. Defaults to "".
        senders (list, optional): mark the sender heads red . Defaults to [].
        print_scores (bool, optional): print the influence scores in the heatmap. Defaults to True.
    """
    n_layers, n_heads = scores.shape
    scores =  scores.to("cpu")


    fig, ax = plt.subplots()

    # if a value is Nan, the color will be grey
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='lightgray')  # NaN will appear light grey
    
    # plot
    im = plt.pcolormesh(scores, cmap=cmap, norm=mpl.colors.CenteredNorm(), edgecolors='k', linewidth=0.5)
    
    divider = make_axes_locatable(ax)
    cbar_width = 0.05 if n_heads <= 12 else 0.03 if n_heads <= 32 else 0.02
    
    cax = divider.append_axes("right", size=f"{cbar_width*100}%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(color_axis_title, rotation=-90, va="bottom", fontsize=cbar_fontsize)
    
    # set the height and width
    fig_height = max(6, n_layers * 0.5)
    fig_width = max(7, n_heads * 0.5)
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    # title
    ax.set_title(title, fontsize=title_font)

    # axis: ticks and lables
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("Heads",  color="black", fontsize=fontsize)
    ax.set_xticks(np.arange(0.5, scores.shape[1], 1))
    ax.set_xticklabels(np.arange(0, scores.shape[1], 1), fontsize=labelsize)

    ax.set_ylabel("Layers",  color="black", fontsize=fontsize)
    ax.set_yticks(np.arange(0.5, scores.shape[0], 1))
    ax.set_yticklabels(np.arange(0, scores.shape[0], 1),  fontsize=labelsize)
    
    # show the values on the heatmap    
    if print_scores:
        for i in range(n_layers):
            for j in range(n_heads):
                #print(logits_diff[i,j])
                if scores[i,j].isnan():
                    continue
                else:
                    ax.text(j+0.5, i+0.5, f"{scores[i,j].tolist():.3f}",
                        ha="center", va="center", fontsize=val_size)

    
    # if sender nodes not none, mark the picked senders
    for s in senders:
        ax, handles = outline_senders(node=s, ax=ax)
        
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        save_img(fig, subfolder, name+ ".png")


def heat_map_layer_pos(
    metric_scores: Float[Tensor, "layer pos"], 
    title: str, 
    color_axis_title: str, 
    show: bool = True,
    save: bool = False,
    name: str = "",
    subfolder: str = "",
    labels: Optional[str] = None
    ):
    """Heatmap (Layer x Seq_Pos) 

    Args:
        metric_scores ( Float[Tensor, "layer head"]): Influence scores for each head over a specific metric (usually avg_logit_diff)
        title (str): _description_
        color_axis_title (str): _description_
        show (bool, optional): _description_. Defaults to True.
        save (bool, optional): _description_. Defaults to False.
        name (str, optional): _description_. Defaults to "".
        subfolder (str, optional): _description_. Defaults to "".
        labels (Optional[str], optional): _description_. Defaults to None.
    """
    n_layers, seq_len = metric_scores.shape
    
    fig, ax = plt.subplots()
    im = ax.imshow(metric_scores, cmap="RdBu", norm=mpl.colors.CenteredNorm())
    
    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(0, n_layers))
    ax.set_xticks(np.arange(0, seq_len))

    if labels is not None:
        ax.set_xticklabels(labels)
        ax.xaxis.set_tick_params(rotation=45, labelsize=10)
    
    ax.set_ylabel("Layers")
    ax.set_xlabel("Position")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(color_axis_title, rotation=-90, va="bottom")

    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        save_img(fig, subfolder, name+ ".png")


def heat_map_pruning(
    scores: Float[Tensor, "layer head"], 
    GT_CIRCUIT:dict, 
    PRUNING_CIRCUIT:dict, 
    title:str="heatmap_pruning_scores", 
    title_pruning_circuit:str="Pruning Circuit",
    title_gt_circuit:str="GT Circuit",
    title_temp_scale:str="Pruning score",
    performance:float=None, 
    subtitle:str=None, 
    print_scores:bool=True,
    scale_on:bool=True, 
    print_text:bool=True
    ):
    """Heatmap (layer x head) over the pruning scores. Different metrics (WIFN, IFV, WIFV) for FLAP. 
    Heads in the GT circuit, in the FLAP circuit and in the Union of these circuits are marked in different colours.
 
    Args:
        scores (Float[Tensor, "layer head"]): pruning scores
        GT_CIRCUIT (dict): ground truth circuit (prior found via path patching)
        PRUNING_CIRCUIT (dict): Pruning circuit
        title (str, optional): title. Defaults to "heatmap_pruning_scores".
        title_pruning_circuit (str, optional): title of the pruning circuit (e.g. FLAP Circuit). Defaults to "Pruning Circuit".
        title_gt_circuit (str, optional): title of the gt circuit (e.g IOI Circuit). Defaults to "GT Circuit".
        title_temp_scale (str, optional): title of the metric. Defaults to "Pruning score".
        performance (float, optional): If not None, printed next to the heatmap. Defaults to None.
        subtitle (str, optional): subtitle. Defaults to None.
        print_vals (bool, optional): If True, print pruning scores. Defaults to True.
        scale_on (bool, optional): If True, scale figure size with number of heads and layers. Defaults to True.
        print_text (bool, optional): If True, print TPR, circuit_size information next to the heatmap. Defaults to True.

    Returns:
        _type_: _description_
    """    
    legend_anchor_y = 1 + (1 / n_layers)
    legend_anchor_x = -0.5
    
    n_layers, n_heads = scores.shape
    fig, ax = plt.subplots()

    # plot
    if scale_on:
        im = plt.pcolormesh(scores, cmap="RdBu", edgecolors='k', linewidth=0.5, norm=mpl.colors.CenteredNorm())
        divider = make_axes_locatable(ax)
        cbar_width = 0.05 if n_heads <= 12 else 0.03 if n_heads <= 32 else 0.02
        
        cax = divider.append_axes("right", size=f"{cbar_width*100}%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(title_temp_scale, rotation=-90, va="bottom", fontsize=cbar_fontsize, color="black")
        cbar.ax.tick_params(colors="black") 
    else: 
        im = plt.pcolormesh(scores, cmap="Greys", edgecolors='k', linewidth=0.5, norm=mpl.colors.CenteredNorm())
    
    # set the height and width
    fig_height = max(7, n_layers * 0.75)
    fig_width = max(6, n_heads * 0.75)
    
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    
    # title
    if not subtitle is None:
        fig.suptitle(title, horizontalalignment='center')
        ax.set_title(subtitle, fontsize=title_font)
    else:
        ax.set_title(title, fontsize=title_font)

    # axis:
    ax.invert_yaxis()
    ax.set_aspect("equal")
    
    ax.set_ylabel("Layers", color="black", fontsize=fontsize)
    ax.set_xlabel("Heads", color="black", fontsize=fontsize)
    
    # axis: ticks and lables
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    ax.set_yticks(np.arange(0.5, scores.shape[0], 1))
    ax.set_yticklabels(np.arange(0,  scores.shape[0], 1),  fontsize=labelsize)

    
    ax.set_xticks(np.arange(0.5, scores.shape[1], 1))
    ax.set_xticklabels(np.arange(0,  scores.shape[1], 1), fontsize=labelsize)

    # show the values on the heatmap
    if print_scores:
        for i in range(n_layers):
            for j in range(n_heads):
                ax.text(j, i, f"{scores[i,j].tolist():.1f}",
                        ha="center", va="center", fontsize=val_size)
                
    # outline heads in circuits:
    ax, handles = outline_heads(scores, GT_CIRCUIT, PRUNING_CIRCUIT, ax, title_test_circuit=title_pruning_circuit, title_gt_circuit=title_gt_circuit)
    
    # add TPR, FPR, size and performance box next to the plot if print_text=true
    if print_text:
        text_anchor_x = legend_anchor_x 
        text_anchor_y = legend_anchor_y - 0.15 - (1/ n_layers)
    
        text = f""" TPR: {TPR(PRUNING_CIRCUIT, GT_circuit=GT_CIRCUIT)*100:.2f} %
                    \n FPR: {FPR(PRUNING_CIRCUIT, GT_circuit=GT_CIRCUIT)*100:.2f} %
                    \n size {title_pruning_circuit} : {circuit_size(PRUNING_CIRCUIT)}"""
        if not performance is None:
            text = text + f"\n \n performance: {performance:.2f}%"
    
        plt.text(
            x=text_anchor_x, #* n_heads,
            y=text_anchor_y, #* n_layers,
            transform=ax.transAxes,
            s= text,
            fontsize= fontsize ,#+ ((n_layers / 12) - 1),
            verticalalignment='top',
            bbox=dict(facecolor="none", edgecolor='black', boxstyle='round,pad=0.')
            )

    # legend
    plt.legend(
        handles=handles, 
        loc="upper left",
        bbox_to_anchor=(legend_anchor_x, legend_anchor_y),
        frameon=True, 
        bbox_transform=ax.transAxes, 
        fontsize=fontsize
        )
   
    fig.tight_layout()
    return fig


#----------------------------------------------------------------------------------------------------
# Outlines
#---------------------------------------------------------------------------------------------------_

def outline_senders(node, ax, color="red", label=""):
    """Outline node in red"""
    rect = mpl.patches.Rectangle((node[1], node[0]), 1, 1, linewidth=2, edgecolor=color, facecolor='none', label=label)
    ax.add_patch(rect)
    return ax, rect


def outline_heads(
    scores: Float[Tensor, "layer head"],
    GT_CIRCUIT:dict, 
    TEST_CIRCUIT:dict,
    ax,
    title_test_circuit:str="Pruning Circuit", 
    title_gt_circuit:str="GT Circuit"
    ):
    """Retrun differently coloured outlines for nodes in GT_CIRCUIT, TEST_CIRCUIT and UNION(GT_CIRCUIT, TEST_CIRCUIT) i

    Args:
        scores (Float[Tensor, "layer head"]): pruning scores
        GT_CIRCUIT (dict): ground truth circuit
        TEST_CIRCUIT (dict): evaluated circuit
        ax (): axis of a plot
        title_test_circuit (str, optional): title of test circuit. Defaults to "Pruning Circuit".
        title_gt_circuit (str, optional): titke of ground truth circuit. Defaults to "GT Circuit".

    Returns:
        ax, handles: axis and list of rectangles 
    """
    
    rect_UNION=None
    rect_EVAL=None
    rect_GT=None
    linewidth=2.5
    for layer in range(scores.shape[0]):
        for head in range(scores.shape[1]):
            in_GT = head in GT_CIRCUIT.get(layer) if GT_CIRCUIT.get(layer) is not None else False
            in_EVAL = head in TEST_CIRCUIT.get(layer) if TEST_CIRCUIT.get(layer) is not None else False
            if in_GT and in_EVAL:            
                rect_UNION = mpl.patches.Rectangle((head , layer ), 1, 1, linewidth=linewidth, edgecolor='#55a868', facecolor='none', label="Union")
                ax.add_patch(rect_UNION)            
            elif in_EVAL:
                rect_EVAL = mpl.patches.Rectangle((head , layer ), 1, 1, linewidth=linewidth, edgecolor="#8172b3", facecolor='none', label=title_test_circuit)
                ax.add_patch(rect_EVAL)        
            elif in_GT:
                rect_GT = mpl.patches.Rectangle((head, layer), 1, 1, linewidth=linewidth, edgecolor='#cfcf1f', facecolor='none', label=title_gt_circuit)
                ax.add_patch(rect_GT)

    
    handles = rect_UNION, rect_EVAL, rect_GT
    handles = [h for h in handles if h is not None]
    return ax, handles
       
#----------------------------------------------------------------------------------------------------
# Evaluation Metrics vs Sparsity
#----------------------------------------------------------------------------------------------------

def choose_metric_sparsity_plot_function(
    df1:pd.DataFrame, 
    cliff_value1,  
    df2:pd.DataFrame=None,
    cliff_value2=None,
    y_metric1:str="Performance", 
    y_metric2:str=None, 
    title="", 
    p1=None, 
    p2=None
):
    if df2 is None:
        if y_metric2 is None:
            one_curve_one_metric_sparsity(
                df=df1, 
                cliff_value=cliff_value1, 
                y_metric=y_metric1, 
                title=title)
        else:
            one_curve_multiple_metrics_sparsity(
                df=df1, 
                cliff_value=cliff_value1, 
                y_metric1=y_metric1,
                y_metric2=y_metric2
            )
    else:
        if y_metric2 is None:
            two_curves_one_metric_sparsity(
                df1=df1, 
                df2=df2, 
                cliff_value1=cliff_value1, 
                cliff_value2=cliff_value2, 
                y_metric=y_metric1, 
                title=title
            )
        else:
            two_curves_multiple_metrics_sparsity(
                df1=df1, 
                df2=df2, 
                cliff_value1=cliff_value1, 
                cliff_value2=cliff_value2, 
                y_metric1=y_metric1,
                y_metric2=y_metric2, 
                title=title, 
                p1=p1, 
                p2=p2
            )

            
def one_curve_one_metric_sparsity(
    df:pd.DataFrame, 
    cliff_value:Union[int, float], 
    y_metric:str="performance", 
    title=""
    ):
    """Plot one Evaluation Metric vs sparsity ratio

    Args:
        df (pd.DataFrame): df
        cliff_value (Union[int, float]): cliff_value, if int: idx of dataset, if float: sparsity ratio
        y_metric (str, optional): which y value of the df is plotted ["performance", "TPR", "FPR"]. Defaults to "performance".
        title (str, optional): title. Defaults to "".
    """
    
    plt.style.use("ggplot")
    custom_colors = sns.color_palette("Set2")
    plt.rcParams["axes.prop_cycle"] = cycler(color=custom_colors)

    # get important cols from df
    sparsities = df[y_metric]
    y = df[y_metric]
    
    # suplot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.05)
    
    ax.plot(sparsities, y, 's-')
    ax.set_ylim(max(0, min(y) - 2), max(y) + 2)  

    # find the cliff points as index of the df 
    if cliff_value < 1:
        cliff_idx = y.tolist().index(cliff_value)
    else:
        cliff_idx = cliff_value
    ax.axvline(x=sparsities.iloc[cliff_idx], color="red", linestyle=":")
    
    # legends and labels
    axis_label_size=13
    plt.xlabel("Sparsity Ratio", fontsize=axis_label_size)
    ax.set_ylabel("True Positives", fontsize=axis_label_size)
    plt.xticks(rotation=0,  fontsize=axis_label_size)
    plt.yticks(rotation=0,  fontsize=axis_label_size)
    ax.set_title(title)
       
       
def one_curve_multiple_metrics_sparsity(
    df:pd.DataFrame, 
    cliff_value:Union[int, float], 
    y_metric1:str="performance", 
    y_metric2:str="TPR", 
    title=""
    ):
    """Subplot with two evaluation metrics over different sparsity ratios. 
    Usually performance and TPR are the evaluation metrics.

    Args:
        df (pd.DataFrame): dataframe
        y_metric1 (str, optional): evaluation metric 1. Defaults to "performance".
        y_metric2 (str, optional): evaluation metric 2. Defaults to "TPR".
        cliff_value (Union[int, float], optional): Extracted cliff point. Defaults to -1.
        title (str, optional): title. Defaults to "".

    Returns:
        fig: Figure
    """    
    plt.style.use("ggplot")
    custom_colors = sns.color_palette("Set2")
    plt.rcParams["axes.prop_cycle"] = cycler(color=custom_colors)
    
    # get important cols from df
    sparsities = df["sparsity_ratio"]
    y1 = df[y_metric1] # usually performance
    y2 = df[y_metric2] # usually TPR
   
    # subplot

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6),gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.05)

    ax1.plot(sparsities, y1, 'o-')
    ax2.plot(sparsities, y2, 's-')
    
    ax1.set_ylim(min(y1), max(y1)) 
    ax2.set_ylim(min(y2) - 2, max(y2) + 2)  
    
    # Hide the spines between the two plots
    ax2.spines['top'].set_visible(False)

    ax1.tick_params(labeltop=False)  
    ax2.tick_params(labeltop=False)

    # find the cliff points as index of the df 
    if cliff_value < 1:
        cliff_idx = sparsities.tolist().index(cliff_value)
    else:
        cliff_idx = cliff_value
        
    ax1.axvline(x=sparsities.iloc[cliff_idx], color="red", linestyle=":")
    ax2.axvline(x=sparsities.iloc[cliff_idx], color="red", linestyle=":")
    
    # legends and labels
    axis_label_size=13
    plt.xlabel("Sparsity Ratio", fontsize=axis_label_size)
    ax1.set_ylabel("Perfromance (%)", fontsize=axis_label_size)
    ax2.set_ylabel("True Positives", fontsize=axis_label_size)
    plt.xticks(rotation=0,  fontsize=axis_label_size)
    plt.yticks(rotation=0,  fontsize=axis_label_size)
    ax1.set_title(title)

    return fig


def two_curves_one_metric_sparsity(
    df1:pd.DataFrame, 
    df2:pd.DataFrame,  
    cliff_value1:Union[int, float], 
    cliff_value2:Union[int, float], 
    y_metric:str="performance",
    title:str=""
    ):
    
    """Plot two curves from two different circuits with two different cliff points over one metric """
    plt.style.use("ggplot")
    custom_colors = sns.color_palette("Set2")
    plt.rcParams["axes.prop_cycle"] = cycler(color=custom_colors)

    # get important cols from df
    curve1_sparsities = df1["sparsity_ratio"]
    curve2_sparsities = df2["sparsity_ratio"]
    curve1_y = df1[y_metric]
    curve2_y = df2[y_metric]
    
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.05)

    line1, = ax.plot(curve1_sparsities, curve1_y, 's-', label="Vanilla", linewidth=2)
    line2, = ax.plot(curve2_sparsities, curve2_y, 's-', label="Subtracted", linewidth=2)
    
    ax.set_ylim(min(curve1_y) - 2, max(curve1_y) + 2)  

    if cliff_value1 < 1:
        cliff_idx1 = curve1_sparsities.tolist().index(cliff_value1)
    else:
        cliff_idx1 = cliff_value1
    
    ax.axvline(
        x=curve1_sparsities.iloc[cliff_idx1], 
        color=line1.get_color(),
        linestyle=":", 
        linewidth=3
        )
    

    if cliff_value2 < 1:
        cliff_idx2 = curve2_sparsities.tolist().index(cliff_value2)
    else:
        cliff_idx2 = cliff_value2
        
    ax.axvline(
        x=curve2_sparsities.iloc[cliff_idx2],
        color=line2.get_color(),
        linestyle=":", 
        linewidth=3
        )
    
    # legends and labels
    plt.xlabel("Sparsity Ratio", fontsize=title_font)
    ax.set_ylabel("True Positives", fontsize=title_font)
    plt.xticks(rotation=0,  fontsize=title_font)
    plt.yticks(rotation=0,  fontsize=title_font)
    ax.set_title(title)
    plt.legend(loc='upper right', fontsize=title_font)

    return fig


def two_curves_multiple_metrics_sparsity(
    df1:pd.DataFrame, 
    df2:pd.DataFrame,
    cliff_value1,  
    cliff_value2,
    y_metric1:str="Performance", 
    y_metric2:str="TPR", 
    title="", 
    p1=None, 
    p2=None
    ):
    """Plot two evaluation metrics for two different curves to the sparsity ratio"""

    plt.style.use("ggplot")
    custom_colors = sns.color_palette("Set2")
    plt.rcParams["axes.prop_cycle"] = cycler(color=custom_colors)
    
    # get important cols from df
    curve1_sparsities = df1["sparsity_ratio"]
    curve2_sparsities = df2["sparsity_ratio"]

    curve1_metric1 = df1[y_metric1] # usually performance
    curve1_metric2 = df1[y_metric2] # usually TPR
    curve2_metric1 = df2[y_metric1] # usually performance
    curve2_metric2 = df2[y_metric2] # usually TPR

    # subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7),gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.05)

    col2 = "#348246"
    col_p1 = "#00e20f"
    col1 = "#8172b3"
    col_p2="#dd48fb"

    
    ax1.plot(curve1_sparsities, curve1_metric1,'o-', color=col1)
    ax2.plot(curve1_sparsities,  curve1_metric2, 's-', color=col1,)
    
    ax1.plot(curve2_sparsities, y_metric2, 'o-', color=col2)
    ax2.plot(curve2_sparsities, curve2_metric2, 's-', color=col2)
    
    ax1.set_ylim(min(min(curve1_metric1), min(curve2_metric1)), max(max(curve1_metric1), max(curve2_metric1)))
    ax2.set_ylim(min(min(curve1_metric2), min(curve2_metric2)) - 2, max(max(curve1_metric2), max(curve2_metric2)) + 2)  
    
    # Hide the spines between the two plots
    ax2.spines['top'].set_visible(False)

    ax1.tick_params(labeltop=False)  
    ax2.tick_params(labeltop=False)

    # find the cliff points as index of the df 
    if cliff_value1 is not None and cliff_value2 is not None:
        if cliff_value1 < 1:
            cliff_idx1 = curve1_sparsities.tolist().index(cliff_value1)
        else:
            cliff_idx1 = cliff_value1
        
        if cliff_value2 < 1:
            cliff_idx2 = curve2_sparsities.tolist().index(cliff_value2)
        else:
            cliff_idx2 = cliff_value2
        
        # plot cliff points
        ax1.axvline(x=curve1_sparsities.iloc[cliff_idx1], linestyle=":", color=col1,  linewidth=3)
        ax2.axvline(x=curve2_sparsities.iloc[cliff_idx1],  linestyle=":", color=col1,  linewidth=3)
        
        ax1.axvline(x=curve1_sparsities.iloc[cliff_idx2], linestyle=":", color=col2,  linewidth=3)
        ax2.axvline(x=curve2_sparsities.iloc[cliff_idx2],  linestyle=":", color=col2, linewidth=3)


    # if specific sparsity ratios are given, plot those
    if p1 is not None:
        y1 = df1.loc[curve1_sparsities == p1, 'performance'].values[0]
        y2 = df1.loc[curve1_sparsities == p1, 'TPR'].values[0]
        ax1.plot([p1], [y1], color=col_p1, marker='o', markersize=10, linewidth=2, zorder=5)
        ax2.plot([p1], [y2], color=col_p1, marker='o', markersize=10, linewidth=2, zorder=5)

    if p2 is not None:
        y1 = df2.loc[curve2_sparsities == p2, 'performance'].values[0]
        y2 = df2.loc[curve2_sparsities == p2, 'TPR'].values[0]
        ax1.plot([p2], [y1], color=col_p2, marker='o', markersize=10, linewidth=2, zorder=5)
        ax2.plot([p2], [y2], color=col_p2, marker='o', markersize=10, linewidth=2, zorder=5)

    var1_handles = [Line2D([0], [0], color=col1, linestyle='-', label="Contrastive FLAP",  linewidth=3)]
    var2_handles = [Line2D([0], [0], color=col2 , linestyle='-', label="FLAP",  linewidth=3)]
    cliff_handles = []
    point_handle = []
    if cliff_value1 is not None and cliff_value2 is not None:
        cliff_handles = [Line2D([0], [0], color="black" , linestyle=':', linewidth=3, label="Cliff Points")]
    if p1 is not None and p2 is not None:
        point_handle = [Line2D([0], [0], color="black" ,marker='o',markersize=10, label="Fixed Points")]
        
    # Combine all with headers
    all_handles =  var1_handles + var2_handles + cliff_handles + point_handle

    # legends and labels
    ax1.legend(handles=all_handles, loc='lower left', frameon=True, fontsize=fontsize)# bbox_to_anchor=(1.35, 1), fontsize=fontsize)
    plt.xlabel("Sparsity Ratio", fontsize=fontsize)
    ax1.set_ylabel("Perfromance (%)", fontsize=fontsize)
    ax2.set_ylabel("True Positives", fontsize=fontsize)
    ax1.tick_params(axis='x', rotation=0, labelsize=fontsize)
    ax1.tick_params(axis='y', rotation=0, labelsize=fontsize)
    ax2.tick_params(axis='x', rotation=0, labelsize=fontsize)
    ax2.tick_params(axis='y', rotation=0, labelsize=fontsize)
    ax1.set_title(title, fontsize=title_font)

    return fig

    return fig

#----------------------------------------------------------------------------------------------------
# Circuit Analysis - Activation Scores
#----------------------------------------------------------------------------------------------------

def plot_activations(
    activations: Float[Tensor, "seq seq"],
    tokens: Float[Tensor, "1 seq"], 
    head:tuple,
    print_vals:bool=False,
    title:str=""):
    """Plot the Activation Pattern of a head over a specific token sequence during a forward pass

    Args:
        activations (Float[Tensor, "seq seq"]): _description_
        tokens (TFloat[Tensor, "1, seq"]): tokens of one specific sequence 
        head (Tuple[int, int]): (layer_pos, head_pos) of observed head
        print_vals (bool, optional): print activation scores. Defaults to False.
        title (str, optional): title. Defaults to "".

    Returns:
        fig: fig
    """
    # activation matrx in clean activations
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(activations,
                xticklabels=tokens,
                yticklabels=tokens, 
                cmap="viridis",
                vmax=1, vmin=0,
                mask=np.isnan(activations),cbar=True, square=True, linewidths=0.5, linecolor='white')
    
    
    n_layers, n_heads = activations.shape
    if print_vals:
        for i in range(n_layers):
            for j in range(n_heads):
                if np.isnan(activations[i,j]):
                    break
                ax.text(j, i, f"{activations[i,j].tolist():.2f}",
                        ha="left", va="top")
            
    plt.title(f"{title} Layer {head[0]}, Head {head[1]}",  fontsize=title_font)
    #plt.title(f"{title}",  fontsize=axis_label_size + 2)

    plt.xlabel("Attended Token", fontsize=fontsize)
    plt.ylabel("Current Token", fontsize=fontsize)
    plt.xticks(rotation=90,  fontsize=labelsize)
    plt.yticks(rotation=0,  fontsize=labelsize)
    plt.tight_layout()
    return fig
    

#----------------------------------------------------------------------------------------------------
# Circuit Analysis - Barplot of Included Heads
#----------------------------------------------------------------------------------------------------

def circuit_analysis_barplot(
    df: pd.DataFrame, 
    order: dict):
    
    plt.style.use("seaborn-v0_8")

    barplot = df.pivot(index="Label", columns="Method", values="Value").reindex(order)
    ax = barplot.plot(kind="bar", width=0.8, figsize=(8,5), color=['#cfcf1f', '#8172b3', "#E69F00"])

    # title and axis label
    ax.set_title("", fontsize=title_font)
    ax.set_ylabel("Number of Heads", fontsize=fontsize)
    ax.set_xlabel("")

    # x-ticks
    ax.set_xticklabels(barplot.index, rotation=45, ha="right", fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize) 
    ax.tick_params(axis='x',  which="both", length=5, labelrotation=45)

    # legend
    ax.legend(fontsize=fontsize, facecolor="white", frameon=True)
    
    # grid
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis='y', color='white', linewidth=1.2)    

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()
    

#----------------------------------------------------------------------------------------------------
# Circuit Analysis - Pareto Points
#----------------------------------------------------------------------------------------------------

def pareto(
    df:pd.DataFrame,
    x_metric:str,
    y_metric:str,
    all_tasks:List[str], 
    min_performance:int=75, 
    total_model_heads:int=144, 
    save_image:bool=False, 
    show_image:bool=True,
    out_path:str=""):
    """Pareto frontier between two metrics

    Args:
        df (pd.DataFrame): df
        x_metric (str) columm name of df,
        y_metric (str): column name of df,
        all_tasks (List[str]): list of tasks ["ioi", "GreaterThan", "induction", "GenderedPronouns", "Docstring"]
        min_performance (int, optional): minimal performance. Defaults to 75.
        total_model_heads (int, optional): total number of heads in model. Defaults to 144.
        save_image (bool, optional): save_img. Defaults to True.
        out_path (str, optional): out_path. Defaults to "".

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    
    for task in all_tasks:
        df_task = df[df["task"]== task] 

        def pareto_frontier(df, x_metric, y_metric, maximize_y=True, minimize_x=True):
            df_sorted = df.sort_values(by=[x_metric], ascending=minimize_x)
            pareto = []
            best_y = -float("inf") if maximize_y else float("inf")
            for _, row in df_sorted.iterrows():
                y = row[y_metric]
                if maximize_y:
                    if y > best_y:
                        pareto.append(row)
                        best_y = y
                else:
                    if y < best_y:
                        pareto.append(row)
                        best_y = y
            return pd.DataFrame(pareto)

        pareto = pareto_frontier(df_task, x_metric, y_metric)

        try:
            best_point = df_task[df_task["performance"] >= min_performance].sort_values(by="size").iloc[0]
            if best_point["size"] > total_model_heads/2:
                raise Exception
        except:
            best_point = df_task[df_task["performance"] >= df_task["performance"].mean()].sort_values(by="size").iloc[0]


        fig, ax = plt.subplots(figsize=(8,6))
        plt.scatter(df_task["size"], df_task["performance"], label="All Hyperparmetres", alpha=0.6)
        plt.plot(pareto["size"], pareto["performance"], color="red", linewidth=2, label="Pareto front")
        plt.scatter(best_point["size"], best_point["performance"],
                    color="green", s=120, marker="o", edgecolors="black", zorder=5,
                    label=f"Pareto Point\n(size={best_point['size']}, perf={best_point['performance']:.2f})")

        plt.xlabel("Circuit Size")
        plt.ylabel("Performance")
        plt.title("Pareto Frontier: " + task)
        plt.legend()
        plt.grid(True)
        if show_image:
            plt.show()
        
        if save_image:
            save_img(fig, out_path=out_path, name=f"pareto_{task}.png")
            
            
#----------------------------------------------------------------------------------------------------
# Efficency
#----------------------------------------------------------------------------------------------------

def efficency_barplots(
    df:pd.DataFrame, 
    model_name:str,
    efficency_metric:str, 
    save_image:bool=False, 
    save_txt:bool=False, 
    show_image:bool=True, 
    out_path:str=""
    ):
    
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = range(len(df))
    bars_pp = ax.bar([i - width/2 for i in x], df['PP'], width=width, label='PP')
    bars_app = ax.bar([i + width/2 for i in x], df['APP'], width=width, label='APP')

    # Add vertical brackets with horizontal ticks
    tick_width = 0.1  # width of the horizontal ticks
    for i, speedup in enumerate(df[efficency_metric]):
        y_pp = df['PP'][i]
        y_app = df['APP'][i]
        x_pos = i + width/2  # center at APP bar
        y_bottom = min(y_pp, y_app)
        y_top = max(y_pp, y_app)
        
        # Draw vertical line
        ax.plot([x_pos, x_pos], [y_bottom, y_top], color='black', linewidth=1.2)
        
        # Draw horizontal ticks at top and bottom
        ax.plot([x_pos - tick_width/2, x_pos + tick_width/2], [y_top, y_top], color='black', linewidth=1.2)
        ax.plot([x_pos - tick_width/2, x_pos + tick_width/2], [y_bottom, y_bottom], color='black', linewidth=1.2)
        
        # Add speedup text above the bracket
        ax.text(x_pos, y_top*1, f"x{speedup:.2f}", ha='center', va='bottom', fontsize=labelsize)

    # Styling
    ax.set_title(model_name, fontsize=title_font, weight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=0, ha='center')
    ax.tick_params(axis='x', labelsize=labelsize)  # font size for x-axis tick labels
    ax.tick_params(axis='y', labelsize=labelsize)  # font size for y-axis tick labels
    if efficency_metric == "comp_time":
        ax.set_ylabel("Computation Time (sec)",  fontsize=fontsize)
    else:
        ax.set_ylabel(efficency_metric,  fontsize=fontsize)
    ax.set_facecolor('#EAF2F8')
    ax.grid(True, axis='y', color='white', linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.legend(loc='upper right', title=None, fontsize=fontsize)
    plt.tight_layout()
    
    if show_image:
        plt.show()

    if save_image:
        save_img(fig,  f"{out_path}/{model_name}", name="speedup.png")
    if save_txt:
        store_df(df, out_path=f"{out_path}/{model_name}", name="sppedup.json")        