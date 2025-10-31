import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from utils.utils import save_img
from utils.eval_circuit import get_union_num, get_intersection_num, IoU_nodes, circuit_size
from Pruning.OWL.lib.prune_all_hooked import sublayer_statistics, sparsity_ratio
import numpy as np
import matplotlib.patches as patches
import matplotlib as mpl


def IoU_for_parameters(data):
    plot = plt.bar(list(data.keys()), list(data.values()))
    return plot
        
def line_graph(df: pd.DataFrame, df_IOI=None):
 
    fontsize=15
    labelsize=12
 
    if df.shape[1] == 3:
        group_name, x_name, y_name = list(df.columns.values)
    elif df.shape[1] == 4:
        group_name, x_name, y_name, y_name_right = list(df.columns.values)
    else:
        raise Exception(f"Dimension of df are of: {df.shape[1]}. Should be 3 or 4")
   
    handles, labels = [], []
 
    unique_groups = df[group_name].unique()
    base_linestyles = ["dashed", "dashdot", "dotted"]
    base_pointstyles = ["o", "v", "s", "P"]
    num_groups = len(unique_groups)
    linestyles = [base_linestyles[i % len(base_linestyles)] for i in range(num_groups)]
    pointstyles = [base_pointstyles[i % len(base_pointstyles)] for i in range(num_groups)] 

    fig, ax1 = plt.subplots()

    if not df_IOI is None:
        line = ax1.axhline(y=df_IOI[y_name].values[0], linestyle="solid", label="IOI", color="red")
        if df.shape[1] == 4:
            line = ax1.axhline(y=df_IOI[y_name_right].values[0], linestyle="solid", label="IOI", color="blue")
    
    for i, group in enumerate(df[group_name].unique()):
        line = plt.Line2D([0], [0], marker=pointstyles[i] , linestyle=linestyles[i], label=group, color="black")
        if group not in labels:
            handles.append(line)
            labels.append(group)
    handles.append(plt.Line2D([0], [0], marker=pointstyles[i] , linestyle="solid", label=group, color="black"))
    labels.append("IOI")
    
    for i, group in enumerate(df[group_name].unique()):        
        subset = df[df[group_name] == group]
        line, = ax1.plot(subset[x_name], subset[y_name], marker=pointstyles[i], linestyle=linestyles[i], label=group,color="red")

    ax1.set_ylabel(y_name,  color="red", fontsize=fontsize)
    ax1.tick_params(axis="y", labelcolor="red")
    
    if df.shape[1] == 4:
        ax2 = ax1.twinx()

        for i, group in enumerate(df[group_name].unique()):        
            subset = df[df[group_name] == group]
            linestyle = linestyles[i] 
            line, = ax2.plot(subset[x_name], subset[y_name_right], marker=pointstyles[i] , linestyle=linestyles[i], label=group, color="blue")
        ax2.set_ylabel(y_name_right, color="blue", fontsize=fontsize)
        ax2.tick_params(axis="y", labelcolor="blue", labelsize=labelsize)

    x_labels = df[x_name].unique()
    plt.xticks(ticks=x_labels, labels=x_labels)

    ax1.set_xlabel(x_name, fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    fig.tight_layout()

    fig.legend(handles, labels, title=group_name, loc="upper right",bbox_to_anchor=(1.15, 1))
    #plt.show()
    return fig

def all_figs_from_df(df, name, x_group, y_group, label_group, use_IOI=True):
    
    if use_IOI:
        df_IOI = df[df["circuit_type"] == "IOI"]
    else:
        df_IOI =  None

    if len(y_group) == 1:
        y_name = y_group[0]
        y_name_right = None
    elif len(y_group) == 2:
        y_name = y_group[0]
        y_name_right = y_group[1]
     
    for x_name in x_group:
        for group_name in label_group:
            df_agg = df.groupby([group_name, x_name]).mean().reset_index()
                     
            if y_name_right == None:
                fig = line_graph(df_agg[[group_name, x_name, y_name]], df_IOI=df_IOI)
            else: 
                fig = line_graph(df_agg[[group_name, x_name, y_name, y_name_right]], df_IOI=df_IOI)
                
            save_img(fig, "/mnt/qb/home/eickhoff/esx670/OverlapMetric/res/OWL/", name=f"{name}_{x_name}_{group_name}.png")            
            
            
            
def get_max_row_per_group(df, groups, y_val):
    max_df = pd.DataFrame()
    for group_name in groups:
        for y_name in y_val:
            max_val = df.groupby(group_name)[y_name].max().reset_index()
            max_rows = df.merge(max_val, on=[group_name, y_name], how="inner")
            max_rows["group_by"] = group_name
            max_df = pd.concat([max_df, max_rows], ignore_index=True)
    return max_df.drop_duplicates()

def get_min_row_per_group(df, groups, y_val):
    min_df = pd.DataFrame()
    for group_name in groups:
        for y_name in y_val:
            min_val = df.groupby(group_name)[y_name].min().reset_index()
            min_rows = df.merge(min_val, on=[group_name, y_name], how="inner")
            min_rows["group_by"] = group_name
            min_df = pd.concat([min_df, min_rows], ignore_index=True)
    return min_df.drop_duplicates()