import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch as t
from torch import Tensor
from jaxtyping import Float
import os
from datetime import datetime
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import matplotlib as mpl
from utils.eval_circuit import get_intersection_num, get_union_num, IoU_nodes, circuit_size, TPR, FPR
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import itertools
from matplotlib.lines import Line2D
from cycler import cycler
from cycler import cycler

title_font=16
fontsize=14
labelsize =12



def print_pretty(circle_heads:dict):
    for (key, value) in circle_heads.items():
        print(f"{key} : {value}")
        
# Save Images
def create_folder_structure():
    parent_path = os.path.dirname(os.path.normpath(os.getcwd()))
    result_folder = os.path.join(parent_path, "res")
    print("result folder", result_folder)
    if (not os.path.exists(result_folder)):
        os.makedirs(result_folder)
    return result_folder

def save_image(fig, name:str, subfolder: str = ""):
    if (not os.path.exists(subfolder)):
        print("make dir", subfolder)
        os.makedirs(subfolder)
    
    name = "_".join(name.split(" "))
    fname = os.path.join(subfolder, name + ".png")
    fig.savefig(fname)
    print(f"saved at {fname}")
    
# Graphs for Path Pathcing and Attention Patching   
def heat_map_layer_head(
    logits_diff: Float[Tensor, "layer head"], 
    title: str, 
    color_axis_title: str, 
    show: bool = True,
    save: bool = False,
    name: str = "",
    subfolder: str = "",
    senders: list = [], 
    print_vals=True, 
):
    n_layers, n_heads = logits_diff.shape
    logits_diff =  logits_diff.to("cpu")


    fig, ax = plt.subplots()

    # if a value is Nan, the color will be grey
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='lightgray')  # NaN will appear light grey
    
    
    im = ax.imshow(logits_diff, cmap=cmap, norm=mpl.colors.CenteredNorm())
    
    divider = make_axes_locatable(ax)
    cbar_width = 0.05 if n_heads <= 12 else 0.03 if n_heads <= 32 else 0.02
    
    cax = divider.append_axes("right", size=f"{cbar_width*100}%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(color_axis_title, rotation=-90, va="bottom", fontsize=10)
    
    # set the height and width
    fig_height = max(6, n_layers * 0.5)
    fig_width = max(7, n_heads * 0.5)
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    
    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(0, n_layers))
    ax.set_xticks(np.arange(0, n_heads))
    ax.set_ylabel("Layers")
    ax.set_xlabel("Heads")
    
    
    # add a grid
    ax.set_xticks(np.arange(n_heads + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_layers + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="darkgrey", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)
    
    # show the values on the heatmap    
    if print_vals:
        for i in range(n_layers):
            for j in range(n_heads):
                #print(logits_diff[i,j])
                if logits_diff[i,j].isnan():
                    continue
                else:
                    ax.text(j, i, f"{logits_diff[i,j].tolist():.3f}",
                        ha="center", va="center")

    
    # if sender nodes not none, mark the picked senders
    for s in senders:
        ax, handles = outline_box(scores=logits_diff, node=s, ax=ax)
        
    fig.tight_layout()
    if show:
        plt.show()
    if save:
        print("save!")
        save_image(fig, name, subfolder)

def heat_map_layer_pos(
    logits_diff: Float[Tensor, "layer pos"], 
    title: str, 
    color_axis_title: str, 
    show: bool = True,
    save: bool = False,
    name: str = "",
    subfolder: str = "",
    labels: Optional[str] = None
    ):
    n_layers, seq_len = logits_diff.shape
    
    fig, ax = plt.subplots()
    im = ax.imshow(logits_diff, cmap="RdBu", norm=mpl.colors.CenteredNorm())
    
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
        save_image(fig, name, subfolder)


# Graphs for Pruning
def heat_map_sparsity(
    sparsities, 
    IOI_CIRCUIT, 
    EVAL_CIRCUIT, 
    title="sparsity ratios", 
    title_eval_circuit="OWL Circuit",
    title_compare_circuit="IOI Circuit",
    title_temp_scale="sparsity ratio",
    performance=None, 
    subtitle=None, 
    print_vals=True,
    scale_on=True, 
    print_text=True
    ):
    
    n_layers, n_heads = sparsities.shape
    fig, ax = plt.subplots()

    if scale_on:
        im = plt.pcolormesh(sparsities, cmap="RdBu", edgecolors='k', linewidth=0.5, norm=mpl.colors.CenteredNorm())
        divider = make_axes_locatable(ax)
        cbar_width = 0.05 if n_heads <= 12 else 0.03 if n_heads <= 32 else 0.02
        
        cax = divider.append_axes("right", size=f"{cbar_width*100}%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(title_temp_scale, rotation=-90, va="bottom", fontsize=10, color="black")
        cbar.ax.tick_params(colors="black") 
    else: 
        im = plt.pcolormesh(sparsities, cmap="Greys", edgecolors='k', linewidth=0.5, norm=mpl.colors.CenteredNorm())
        #im = ax.imshow(sparsities, cmap='Greys', vmin=0, vmax=1)#,  cmap="RdBu",vmin=1, vmax=1) #cmap="gist_heat_r")#, norm=mpl.colors.CenteredNorm())
    
    if not subtitle is None:
        fig.suptitle(title, horizontalalignment='center')
        ax.set_title(subtitle, fontsize=title_font)
    else:
        ax.set_title(title, fontsize=title_font)

    fig_height = max(7, n_layers * 0.75)
    fig_width = max(6, n_heads * 0.75)
    
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    # axis:
    ax.invert_yaxis()
    ax.set_aspect("equal")
    
    ax.set_ylabel("Layers", color="black", fontsize=fontsize)
    ax.set_xlabel("Heads", color="black", fontsize=fontsize)
    
    # ticks, colors and lables
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    ax.set_yticks(np.arange(0.5, sparsities.shape[0], 1))
    ax.set_yticklabels(np.arange(0,  sparsities.shape[0], 1),  fontsize=labelsize)

    
    ax.set_xticks(np.arange(0.5, sparsities.shape[1], 1))
    ax.set_xticklabels(np.arange(0,  sparsities.shape[1], 1), fontsize=labelsize)


    # mark the fields that are picked for circuit:
    ax, handles = outline_IoU(sparsities, IOI_CIRCUIT, EVAL_CIRCUIT, ax, title_eval_circuit=title_eval_circuit, title_compare_circuit=title_compare_circuit)
    text = f""" TPR: {TPR(EVAL_CIRCUIT, GT_circuit=IOI_CIRCUIT)*100:.2f} %
                \n FPR: {FPR(EVAL_CIRCUIT, GT_circuit=IOI_CIRCUIT)*100:.2f} %
                \n size {title_eval_circuit} : {circuit_size(EVAL_CIRCUIT)}"""
    if not performance is None:
        text = text + f"\n \n performance: {performance:.2f}%"

    # show the values on the heatmap
    if print_vals:
        for i in range(n_layers):
            for j in range(n_heads):
                ax.text(j, i, f"{sparsities[i,j].tolist():.1f}",
                        ha="center", va="center")
    

    
    legend_anchor_y = 1 + (1 / n_layers)
    legend_anchor_x = -0.5

    plt.legend(
        handles=handles, 
        loc="upper left",
        bbox_to_anchor=(legend_anchor_x, legend_anchor_y),
        frameon=True, 
        bbox_transform=ax.transAxes, 
        fontsize=fontsize
        )
    text_anchor_x = legend_anchor_x 
    text_anchor_y = legend_anchor_y - 0.15 - (1/ n_layers)
    
    if print_text:
        plt.text(
            x=text_anchor_x, #* n_heads,
            y=text_anchor_y, #* n_layers,
            transform=ax.transAxes,
            s= text,
            fontsize= 10 ,#+ ((n_layers / 12) - 1),
            verticalalignment='top',
            bbox=dict(facecolor="none", edgecolor='black', boxstyle='round,pad=0.')
            )


    fig.tight_layout()
    
    plt.show()
    return fig

def outline_IOI(sparsities, IOI_CIRCUIT, ax):      
    for layer in range(sparsities.shape[0]):
        for head in range(sparsities.shape[1]):
            in_IOI = head in IOI_CIRCUIT.get(layer) if IOI_CIRCUIT.get(layer) is not None else False
            if in_IOI:
                rect_IOI = mpl.patches.Rectangle((head - 0.5, layer - 0.5), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none', label="IOI circuit")
                ax.add_patch(rect_IOI)
    return ax

def outline_box(scores, node, ax, color="red", label=""):
    rect = mpl.patches.Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, linewidth=2, edgecolor=color, facecolor='none', label=label)
    ax.add_patch(rect)
    return ax, rect


def outline_IoU(sparsities, IOI_CIRCUIT, CIRCUIT, ax, title_eval_circuit="OWL Circuit", title_compare_circuit="IOI Circuit"):
    rect_BOTH=None
    rect_OWL=None
    rect_IOI=None
    linewidth=2.5
    for layer in range(sparsities.shape[0]):
        for head in range(sparsities.shape[1]):
            in_IOI = head in IOI_CIRCUIT.get(layer) if IOI_CIRCUIT.get(layer) is not None else False
            in_OWL = head in CIRCUIT.get(layer) if CIRCUIT.get(layer) is not None else False
            if in_IOI and in_OWL:            
                rect_BOTH = mpl.patches.Rectangle((head , layer ), 1, 1, linewidth=linewidth, edgecolor='#55a868', facecolor='none', label="Union")
                ax.add_patch(rect_BOTH)            
            elif in_OWL:
                rect_OWL = mpl.patches.Rectangle((head , layer ), 1, 1, linewidth=linewidth, edgecolor="#8172b3", facecolor='none', label=title_eval_circuit)
                ax.add_patch(rect_OWL)        
            elif in_IOI:
                rect_IOI = mpl.patches.Rectangle((head, layer), 1, 1, linewidth=linewidth, edgecolor='#cfcf1f', facecolor='none', label=title_compare_circuit)
                ax.add_patch(rect_IOI)
            
            """if in_IOI and in_OWL:            
                rect_BOTH = mpl.patches.Rectangle((head , layer ), 1, 1, linewidth=linewidth, edgecolor=) #'green', facecolor='lightgreen', label="Union") #, facecolor='none', label="Union")
                ax.add_patch(rect_BOTH)            
            elif in_OWL:
                rect_OWL = mpl.patches.Rectangle((head , layer ), 1, 1, linewidth=linewidth, edgecolor="red", facecolor='lightpink', label=title_eval_circuit)#, facecolor='none', label=title_eval_circuit) # #"#8172b3", facecolor='none', label=title_eval_circuit)
                ax.add_patch(rect_OWL)        
            elif in_IOI:
                rect_IOI = mpl.patches.Rectangle((head, layer), 1, 1, linewidth=linewidth, edgecolor="blue", facecolor='lightblue', label=title_compare_circuit)#  facecolor='none', label=title_compare_circuit)#facecolor='none', label=title_compare_circuit)
                ax.add_patch(rect_IOI)"""
    
    handles = rect_BOTH, rect_OWL, rect_IOI
    handles = [h for h in handles if h is not None]
    return ax, handles
        
def ROC_curve(results, y_metric, cliff_value, title=""):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6),gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.05)

    ax1.plot(results["sparsity_ratio"], y_metric, 'o-')
    ax2.plot(results["sparsity_ratio"], results["TPR"], 's-')
    
    ax1.set_ylim(min(y_metric), max(y_metric)) 
    ax2.set_ylim(min(results["TPR"]) - 2, max(results["TPR"]) + 2)  
    
    # Hide the spines between the two plots
    ax2.spines['top'].set_visible(False)

    ax1.tick_params(labeltop=False)  
    ax2.tick_params(labeltop=False)

    if cliff_value < 1:
        cliff_idx = results["sparsity_ratio"].tolist().index(cliff_value)
    else:
        cliff_idx = cliff_value
    ax1.axvline(x=results["sparsity_ratio"].iloc[cliff_idx], color="red", linestyle=":")
    ax2.axvline(x=results["sparsity_ratio"].iloc[cliff_idx], color="red", linestyle=":")
    
    # legends and labels
    axis_label_size=13
    plt.xlabel("Sparsity Ratio", fontsize=axis_label_size)
    ax1.set_ylabel("Perfromance (%)", fontsize=axis_label_size)
    ax2.set_ylabel("True Positives", fontsize=axis_label_size)
    plt.xticks(rotation=0,  fontsize=axis_label_size)
    plt.yticks(rotation=0,  fontsize=axis_label_size)
    ax1.set_title(title)

    return fig

def two_ROC_curve(results1, y_metric1, cliff_value1,  results2, y_metric2, cliff_value2, title="", p1=None, p2=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7),gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.05)

    col2 = "#348246"
    col1 = "#8172b3"
    
    ax1.plot(results1["sparsity_ratio"], y_metric1,'o-', color=col1)
    ax2.plot(results1["sparsity_ratio"],  results1["TPR"], 's-', color=col1,)
    
    ax1.plot(results2["sparsity_ratio"], y_metric2, 'o-', color=col2)
    ax2.plot(results2["sparsity_ratio"], results2["TPR"], 's-', color=col2)
    
    ax1.set_ylim(min(min(y_metric1), min(y_metric2)), max(max(y_metric1), max(y_metric2)))
    ax2.set_ylim(min(results1["TPR"]) - 2, max(results1["TPR"]) + 2)  
    
    # Hide the spines between the two plots
    ax2.spines['top'].set_visible(False)

    ax1.tick_params(labeltop=False)  
    ax2.tick_params(labeltop=False)

    if cliff_value1 is not None and cliff_value2 is not None:
        if cliff_value1 < 1:
            cliff_idx1 = results1["sparsity_ratio"].tolist().index(cliff_value1)
        else:
            cliff_idx1 = cliff_value1
        
        if cliff_value2 < 1:
            cliff_idx2 = results2["sparsity_ratio"].tolist().index(cliff_value2)
        else:
            cliff_idx2 = cliff_value2
        
        ax1.axvline(x=results1["sparsity_ratio"].iloc[cliff_idx1], linestyle=":", color=col1,  linewidth=3)
        ax2.axvline(x=results2["sparsity_ratio"].iloc[cliff_idx1],  linestyle=":", color=col1,  linewidth=3)
        
        ax1.axvline(x=results1["sparsity_ratio"].iloc[cliff_idx2], linestyle=":", color=col2,  linewidth=3)
        ax2.axvline(x=results2["sparsity_ratio"].iloc[cliff_idx2],  linestyle=":", color=col2, linewidth=3)
        
    #handle1 = [Line2D([0], [0], color=col1 , linestyle='-', label="Contrastive FLAP")]
    #handle2 = [Line2D([0], [0], color=col2 , linestyle='-', label="Vanilla FLAP")]

    p1_handle = []
    p2_handle=[]
    if p1 is not None:
        col_p1 = "#00e20f"
        y1 = results1.loc[results1['sparsity_ratio'] == p1, 'performance'].values[0]
        y2 = results1.loc[results1['sparsity_ratio'] == p1, 'TPR'].values[0]

        p1_idx =  results1["sparsity_ratio"].tolist().index(p1)
        ax1.plot([p1], [y1], color=col_p1, marker='o', markersize=10, linewidth=2, zorder=5)
        ax2.plot([p1], [y2], color=col_p1, marker='o', markersize=10, linewidth=2, zorder=5)
        #p1_handle = [Line2D([0], [0], color=col_p1 , marker='o', label="Fixed Contrastive FLAP")]

    if p2 is not None:
        col_p2="#dd48fb"
        y1 = results2.loc[results2['sparsity_ratio'] == p2, 'performance'].values[0]
        y2 = results2.loc[results2['sparsity_ratio'] == p2, 'TPR'].values[0]
        ax1.plot([p2], [y1], color=col_p2, marker='o', markersize=10, linewidth=2, zorder=5)
        ax2.plot([p2], [y2], color=col_p2, marker='o', markersize=10, linewidth=2, zorder=5)
        #p2_handle = [Line2D([0], [0], color=col_p2 ,marker='o', label="Fixed Vanilla FLAP")]



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

    ax1.legend(handles=all_handles, loc='lower left', frameon=True, fontsize=fontsize)# bbox_to_anchor=(1.35, 1), fontsize=fontsize)
    # legends and labels
    plt.xlabel("Sparsity Ratio", fontsize=fontsize)
    ax1.set_ylabel("Perfromance (%)", fontsize=fontsize)
    ax2.set_ylabel("True Positives", fontsize=fontsize)
    ax1.tick_params(axis='x', rotation=0, labelsize=fontsize)
    ax1.tick_params(axis='y', rotation=0, labelsize=fontsize)
    ax2.tick_params(axis='x', rotation=0, labelsize=fontsize)
    ax2.tick_params(axis='y', rotation=0, labelsize=fontsize)
    ax1.set_title(title, fontsize=title_font)

    return fig




def TP_curve(results, y_metric, cliff_value, title=""):
    plt.style.use("ggplot")
    custom_colors = sns.color_palette("Set2")
    plt.rcParams["axes.prop_cycle"] = cycler(color=custom_colors)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.05)

    ax.plot(results["sparsity_ratio"], results["TPR"], 's-')
    
    ax.set_ylim(max(0, min(results["TPR"]) - 2), max(results["TPR"]) + 2)  

    if cliff_value < 1:
        cliff_idx = results["TPR"].tolist().index(cliff_value)
    else:
        cliff_idx = cliff_value
    ax.axvline(x=results["sparsity_ratio"].iloc[cliff_idx], color="red", linestyle=":")
    
    # legends and labels
    axis_label_size=13
    plt.xlabel("Sparsity Ratio", fontsize=axis_label_size)
    ax.set_ylabel("True Positives", fontsize=axis_label_size)
    plt.xticks(rotation=0,  fontsize=axis_label_size)
    plt.yticks(rotation=0,  fontsize=axis_label_size)
    ax.set_title(title)

    return fig


def two_TP_curve(results1, results2,  cliff_value1, cliff_value2, title=""):
    plt.style.use("ggplot")
    custom_colors = sns.color_palette("Set2")
    plt.rcParams["axes.prop_cycle"] = cycler(color=custom_colors)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.05)

    line1, = ax.plot(results1["sparsity_ratio"], results1["TPR"], 's-', label="Vanilla", linewidth=2)
    line2, = ax.plot(results2["sparsity_ratio"], results2["TPR"], 's-', label="Subtracted", linewidth=2)


    
    ax.set_ylim(min(results1["TPR"]) - 2, max(results1["TPR"]) + 2)  

    if cliff_value1 < 1:
        cliff_idx1 = results1["sparsity_ratio"].tolist().index(cliff_value1)
    else:
        cliff_idx1 = cliff_value1
    ax.axvline(
        x=results1["sparsity_ratio"].iloc[cliff_idx1], 
        color=line1.get_color(),
        linestyle=":", 
        linewidth=3
        )
    
    
    
    if cliff_value2 < 1:
        cliff_idx2 = results2["sparsity_ratio"].tolist().index(cliff_value2)
    else:
        cliff_idx2 = cliff_value2
    ax.axvline(
        x=results2["sparsity_ratio"].iloc[cliff_idx2],
        color=line2.get_color(),
        linestyle=":", 
        linewidth=3
        )
    
    # legends and labels
    axis_label_size=14
    plt.xlabel("Sparsity Ratio", fontsize=title_font)
    ax.set_ylabel("True Positives", fontsize=title_font)
    plt.xticks(rotation=0,  fontsize=title_font)
    plt.yticks(rotation=0,  fontsize=title_font)
    ax.set_title(title)
    plt.legend(loc='upper right', fontsize=title_font)

    return fig


def plot_scores(scores, IOI_CIRCUIT, title):
    n_layers, n_heads = scores.shape
    fig, ax = plt.subplots()
    im = ax.imshow(scores, cmap="RdBu", norm=mpl.colors.CenteredNorm())#, norm=mpl.colors.CenteredNorm())
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("mean scores", rotation=-90, va="bottom")
    ax.set_yticks(np.arange(0, n_layers))
    ax.set_xticks(np.arange(0, n_heads))
    ax.set_ylabel("Layers")
    ax.set_xlabel("Heads")
    ax.set_title(title)

    ax = outline_IOI(scores, IOI_CIRCUIT, ax)

    # show the values on the heatmap
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f"{scores[i,j].tolist():.1f}",
                    ha="center", va="center")

    fig.set_figheight(6)
    fig.set_figwidth(7)
    plt.show()
    
    
def plot_activations(
    activations,
    tokens, 
    head,
    x_label,
    y_label,
    axis_label_size=20,
    tick_label_size=10, 
    annotation_size=8, 
    print_vals=False,
    title=""):
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
                        ha="left", va="top",  fontsize=annotation_size)
            
    plt.title(f"{title} Layer {head[0]}, Head {head[1]}",  fontsize=axis_label_size + 2)
    #plt.title(f"{title}",  fontsize=axis_label_size + 2)

    plt.xlabel(x_label, fontsize=axis_label_size)
    plt.ylabel(y_label, fontsize=axis_label_size)
    plt.xticks(rotation=90,  fontsize=tick_label_size)
    plt.yticks(rotation=0,  fontsize=tick_label_size)
    plt.tight_layout()
    return fig
    
    
def rand_jitter(values, strength = 0.05):
    return values +  np.random.uniform(-strength, strength, size=len(values))


def pareto_points(
    df_results, 
    var1, 
    var2, 
    y_var="performance",
    title="", 
    out_path=None,
    x_fit=[],
    y_fit=[],
    img_name="pareto_points.png"
    
    ):
    markers = ['o', 's', '^', 'v', 'D', '*', 'P', 'X', '<', '>']
    marker_cycle = itertools.cycle(markers)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    var1_group = df_results[var1].unique()
    if var2 == None:
        var2_group = []
    else:
        var2_group = df_results[var2].unique()

    var1_map = {cg: colors[i % len(colors)] for i, cg in enumerate(var1_group)}
    var2_map = {mg: next(marker_cycle) for mg in var2_group}

    plt.figure(figsize=(8, 6))
    #jitter_val = (max(df_results[y_var]) - min(df_results[y_var])) / 50
    jitter_val=0
    if var2 == None:
        for cg, subset in df_results.groupby([var1]):
            plt.scatter(
                subset['size'],
                rand_jitter(subset[y_var], strength=jitter_val),
                color=var1_map.get(cg[0]),
            )    
    else:
        for (cg, mg), subset in df_results.groupby([var1, var2]):
            plt.scatter(
                subset['size'],
                rand_jitter(subset[y_var], strength=jitter_val),
                color=var1_map[cg],
                marker=var2_map[mg],
            )
        
    # fit a saturation curve
    if not (len(x_fit) == 0 or len(y_fit) == 0):
        plt.plot(x_fit, y_fit, "r-", label=f"pareto")


    var1_latex = var1.replace("_", "-")
    if var2 == None:
        var2_latex = ""
    else:
        var2_latex = var2.replace("_", "-")

    # --- Build grouped legend handles ---
    header1 = Line2D([0], [0], color='none', marker='', linestyle='', label=f"$\\bf{{{var1_latex}}}$")
    header2 = Line2D([0], [0], color='none', marker='', linestyle='', label=f"$\\bf{{{var2_latex}}}$")
    #separator = Line2D([0], [0], color='gray', linestyle='-', label="")

    var1_handles = [
        Line2D([0], [0], color=var1_map[cg], marker='o', linestyle='', label=str(cg))
        for cg in var1_group
    ]
    var2_handles = [
        Line2D([0], [0], color='black', marker=var2_map.get(mg), linestyle='', label=str(mg))
        for mg in var2_group
    ]

    # Combine all with headers
    all_handles = [header1] + var1_handles + [header2] + var2_handles

    plt.legend(handles=all_handles, loc='center left',
               bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.xlabel("circuit size")
    plt.ylabel(y_var)
    plt.title(title)

    if out_path is not None:
        print("saved at", out_path)
        plt.savefig(os.path.join(out_path, img_name),
                    bbox_inches='tight')
    else:
        plt.show()
