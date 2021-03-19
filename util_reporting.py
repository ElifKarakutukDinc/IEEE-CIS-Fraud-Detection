import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")


def df_first_look(df):
    """
    This function gets a Python Pandas dataframe and visualize basic information about the dataframe.
    :param df: Dataframe to be analyze
    :return: This function doesn't return anything.
    """
    try:
        print("First 5 rows of dataframe:\n--------------------------\n", df.head())
        print("")
        print("Last 5 rows of dataframe:\n--------------------------\n", df.tail())
        print("")
        print(
            "Row count of dataframe:\n-----------------------\n",
            df.shape[0],
            "\nColumn count of dataframe:\n--------------------------\n",
            df.shape[1],
        )
        print("")
        print(
            "List of columns in the dataframe:\n---------------------------------\n",
            df.columns.values,
        )
        print("")
        print(
            "Looking NaN values and datatypes of columns in the dataframe:\n--------------------------------------------\n"
        )
        print(df.info())
        print("")

    except Exception as e:
        print("Error at df_first_look function: ", str(e))


def df_descriptive_statistics(df, column_list):
    """
    This function gets a Python Pandas dataframe and list of columns to visualize descriptive statistics about those columns.
    :param df: Dataframe to be analyze
    :param column_list: List of columns to filter out only numeric columns to use in the fuction"
    :return: This function doesn't return anything.

    """
    try:
        dummy_df = df[column_list].copy()
        print(
            f"Descriptive Statisctics for column:\n--------------------------\n",
            dummy_df.describe(),
        )
        print("")
        print(f"Mode values for column:\n--------------------------\n", dummy_df.mode())
        print("")
    except Exception as e:
        print("Error at df_descriptive_statistics function: ", str(e))


def df_pivot_aggregated_statistics(
    df, column_list, to_be_calculated_column, descriptive_statistic_list
):
    """
    This function gets a Python Pandas dataframe and calculating aggregation by columns in the dataframe.
    :param df: Dataframe to be analyze
    :param column_list: It takes a column list.
    :param to_be_calculated_column: Calculation aggregation for this column.
    :param descriptive_statistic_list: It takes a list which we use descriptive statistic(Mean,Median,Standard dev,Min,etc.)
    :return: This function doesn't return anything.

    """
    df_dummy = (
        df.groupby(column_list)[to_be_calculated_column]
        .agg(descriptive_statistic_list)
        .round(2)
    )
    print(df_dummy)


def countplot_viz(
    data,
    xcolumn,
    xlabel,
    ylabel,
    title,
    hue=None,
    fontsize_label=16,
    fontsize_title=20,
    fontsize_text=12,
    rotation=45,
    figsize_x=12,
    figsize_y=5,
    palette="mako",
):
    """
    This function gets a Python Pandas dataframe and visualize a countplot.
    :param data: Dataframe to be analyze
    :param xcolumn: This column designates x axis column.
    :param xlabel: It designates name of x axis column.
    :param ylabel: It designates name of y axis column.
    :param title: This column designates name of graph.
    :param hue: Name of variables in `data` or vector data, optional Inputs for plotting long-form data.
    :param fontsize_label: It designates label size.
    :param fontsize_title: It designates title size.
    :param rotation: It designates rotation of graph.
    :param palette: It designates colors of graph.
    :return: This function doesn't return anything.

    """
    plt.figure(figsize=(figsize_x,figsize_y))
    
    g = sns.countplot(x=xcolumn, data=data, hue=hue, palette=palette)
    g.set_title(title, fontsize=19)
    g.set_xlabel(xlabel, fontsize=17)
    g.set_ylabel(ylabel, fontsize=17)
    g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    for p in g.patches:
        height = p.get_height()
        g.text(
            p.get_x() + p.get_width() / 2.0,
            height + 3,
            "{:1}".format(height),
            ha="center",
            fontsize=fontsize_text,
        )    
    if hue != None:
        g.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)  
        
def multiple_plot_viz(
    data,
    column_for_separate,
    column_for_calculate,
    title_1,
    title_2,
    edgecolor="black",
    color_1="orange",
    color_2="blue",
    bins=20,
):
    """
    This function gets a Python Pandas dataframe and visualize two countplots.
    :param data: Dataframe to be analyze
    :param xcolumn: This column designates x axis column.
    :param xlabel: It designates name of x axis column.
    :param ylabel: It designates name of y axis column.
    :param title_1: This column designates one of graphs' name.
    :param title_2: This column designates one of graphs' name.
    :param edgecolor: It designates egecolor of graphs.
    :param color_1: It designates color of graph_1.
    :param color_2: It designates color of graph_1.
    :param bins: It designates size of histogram bins.
    :return: This function doesn't return anything.

    """
    f, ax = plt.subplots(1, 2, figsize=(15, 5))

    data[data[column_for_separate] == 0][column_for_calculate].plot.hist(
        ax=ax[0], bins=bins, edgecolor=edgecolor, color=color_1
    )
    ax[0].set_title(title_1)
    x1 = list(range(0, 85, 5))
    ax[0].set_xticks(x1)

    data[data[column_for_separate] == 1][column_for_calculate].plot.hist(
        ax=ax[1], bins=bins, edgecolor=edgecolor, color=color_2
    )
    ax[1].set_title(title_2)
    x2 = list(range(0, 85, 5))
    ax[1].set_xticks(x2)

def countplot_pointplot_viz(
    data,
    filter_list,
    xcolumn,
    ycolumn,
    ycolumn_point,
    xlabel,
    ylabel,
    title,
    hue=None,
    fontsize_label=16,
    fontsize_title=20,
    fontsize_text=12,
    rotation=45,
    figsize_x=12,
    figsize_y=5,
    palette="mako",
):
    """
    This function gets a Python Pandas dataframe and visualize a countplot and a pointplot.
    :param data: Dataframe to be analyze
    :param filter_list: It takes conditions for filtering. 
    :param xcolumn: This column designates x axis column.
    :param ycolumn: This column separetes data by its conditions at countplot. 
    :param ycolumn_point: This column separetes data by its conditions at pointplot. 
    :param xlabel: It designates name of x axis column.
    :param ylabel: It designates name of y axis column.
    :param title: This column designates name of graph.
    :param hue: Name of variables in `data` or vector data, optional Inputs for plotting long-form data.
    :param fontsize_label: It designates label size.
    :param fontsize_title: It designates title size.
    :param rotation: It designates rotation of graph.
    :param palette: It designates colors of graph.
    :return: This function doesn't return anything.

    """    
    
    plt.figure(figsize=(figsize_x,figsize_y)) 
    
    filter_list = filter_list
    df2 = data[data[ycolumn].isin(filter_list)]
    order = df2[xcolumn].value_counts().index

    ax1 = sns.countplot(
        x=xcolumn, hue=ycolumn, data=df2, hue_order=filter_list, palette=palette
    )
    for p in ax1.patches:
        height = p.get_height()
        ax1.text(
            p.get_x() + p.get_width() / 2.0,
            height + 3,
            "{:1}".format(height),
            ha="center",
            fontsize=fontsize_text,
        )
    ax1.set_title(title, fontsize=19)
    ax1.set_xlabel(xlabel, fontsize=17)
    ax1.set_ylabel(ylabel, fontsize=17)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    
    ax2 = ax1.twinx()
    sns.pointplot(x=xcolumn, y=ycolumn_point, data=df2, ax=ax2)




def crosstab_viz(data, index_column_1, index_column_2, aggregated_column, cmap="mako"):
    """
    This function gets a Python Pandas dataframe and visualize a crosstab.
    :param data: Dataframe to be analyze
    :param index_column_1: This column works as a stpliter of pivot.
    :param index_column_2: This column works as a stpliter of pivot.
    :param aggregated_column: Calculation aggregation for this column.
    :param cmap: It designates colors of graph.
    :return: This function doesn't return anything.

    """
    return pd.crosstab(
        [data[index_column_1], data[index_column_2]],
        data[aggregated_column],
        margins=True,
    ).style.background_gradient(cmap=cmap)


def factor_plot_viz(
    data, index_column_1, index_column_2, aggregated_column, title_factor
):
    """
    This function gets a Python Pandas dataframe and visualize a factor plot.
    :param data: Dataframe to be analyze
    :param index_column_1: This column works as a stpliter of pivot.
    :param index_column_2: This column works as a stpliter of pivot.
    :param aggregated_column: Calculation aggregation for this column.
    :param title_factor: It designates name of graph.
    :return: This function doesn't return anything.

    """
    plt.figure(figsize=(12, 5))
    sns.factorplot(aggregated_column, index_column_2, hue=index_column_1, data=data)
    plt.title(title_factor)


def relationship_viz(
    data, index_column_1, index_column_2, aggregated_column, title_factor
):
    """
    This function gets a Python Pandas dataframe and concats two different graphs that are factorplot and crosstab.
    :param data: Dataframe to be analyze
    :param index_column_1: This column works as a stpliter of pivot.
    :param index_column_2: This column works as a stpliter of pivot.
    :param aggregated_column: Calculation aggregation for this column.
    :param title_factor: It designates name of factorplot graph.
    :return: This function doesn't return anything.

    """
    factor_plot_viz(
        data, index_column_1, index_column_2, aggregated_column, title_factor
    )
    return crosstab_viz(data, index_column_1, index_column_2, aggregated_column)


def swarmplot_viz(
    df,
    colum_1,
    colum_2,
    groupped_column,
    xlabel,
    ylabel,
    title,
    fontsize=15,
    fontsize_title=17,
    palette="mako",
):
    """
    This function gets a Python Pandas dataframe and visualize a swarmplot for 2 column by a grouping column.
    :param df: Dataframe to be analyze
    :param column_list: list of columns.
    :param groupped_column: Calculation aggregation for this column.
    :param xlabel: It designates name of x axis column.
    :param ylabel: It designates name of y axis column.
    :param title_1: These columns designates name of graph1.
    :param title_2: These columns designates name of graph2.
    :param fontsize: It designates label size.
    :param fontsize_title: It designates title size.
    :param palette: It designates colors of graph.
    :return: This function doesn't return anything.

    """
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 2)

    sns.swarmplot(colum_1, colum_2, data=df, hue=groupped_column, palette=palette)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title, fontsize=fontsize_title)

    plt.subplots_adjust(hspace=0.5, top=0.9)


def boxplot_viz(
    data,
    xcolumn,
    xlabel,
    title,
    hue=None,
    fontsize_label=16,
    fontsize_title=20,
    rotation=45,
    palette="mako",
):
    """
    This function gets a Python Pandas dataframe and visualize a countplot.
    :param data: Dataframe to be analyze
    :param xcolumn: This column shows x axis column.
    :param xlabel: It designates name of x axis column.
    :param ylabel: It designates name of y axis column.
    :param title: This column designates name of graph.
    :param hue: Name of variables in `data` or vector data, optional Inputs for plotting long-form data.
    :param fontsize_label: It designates label size.
    :param fontsize_title: It designates title size.
    :param rotation: It designates rotation of graph.
    :param palette: It designates colors of graph.
    :return: This function doesn't return anything.

    """
    plt.figure(1, figsize=(9, 6))

    sns.boxplot(x=xcolumn, data=data, hue=hue, palette=palette)
    plt.xlabel(xlabel, fontsize=fontsize_label) 
    plt.title(title, fontsize=fontsize_title)
    plt.xticks(rotation=rotation)
    plt.show()


def histogram_viz(
    data,
    column,
    separate_column,
    condition_1,
    condition_2,
    label1,
    label2,
    color1 = None,
    color2 = None,
):
    """
    Gets a Python Pandas dataframe and visualize histogram by a column's conditions.
    :param data: Dataframe to be analyze
    :param column: This column is for showing data distribution.
    :param separate_column: this colum is for creating histogram by a column's conditions.
    :param condition_1: It designates condition of separate column.
    :param condition_2: It designates condition of separate column.
    :param label1: It designates label by condition_1.
    :param label2: It designates label by condition_2.
    :param color1: It designates color for condition_1.
    :param color2: It designates color for condition_2.
    :return: This function doesn't return anything.

    """
    plt.hist(
        list(data[data[separate_column] == condition_1][column]),
        alpha=0.5,
        label=label1,
        color=color1,
            )
    plt.hist(
        list(data[data[separate_column] == condition_2][column]),
        alpha=0.5,
        label=label2,
        color=color2,
    )
    plt.legend(loc="upper right")
    plt.show()

def histogram_multiple_viz(
    data,
    column,
    separate_column,
    condition_1,
    condition_2,
    title1,
    title2,
    title3,
    title4,
    color1="blue",
    color2="darkorange",
):
    """
    Gets a Python Pandas dataframe and visualize four histograms by a column's conditions and by gets its log.
    :param data: Dataframe to be analyze
    :param column: This column is for showing data distribution.
    :param separate_column: this colum is for creating histogram by a column's conditions.
    :param condition_1: It designates condition of separate column.
    :param condition_2: It designates condition of separate column.
    :param title1: It designates title by graph1.
    :param title2: It designates title by graph2.
    :param title3: It designates title by graph3.
    :param title4: It designates title by graph4.
    :param color1: It designates color for condition_1.
    :param color2: It designates color for condition_2.
    :return: This function doesn't return anything.

    """    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 6))
    data.loc[data[separate_column] == condition_1][column].apply(
        np.log
    ).plot(
        kind="hist",
        bins=100,
        title=title1,
        color=color1,
        xlim=(-3, 10),
        ax=ax1,
    )
    data.loc[data[separate_column] == condition_2][column].apply(
        np.log
    ).plot(
        kind="hist",
        bins=100,
        title=title2,
        color=color2,
        xlim=(-3, 10),
        ax=ax2,
    )
    data.loc[data[separate_column] == condition_1][column].plot(
        kind="hist", bins=100, title=title3, color=color1, ax=ax3
    )
    data.loc[data[separate_column] == condition_2][column].plot(
        kind="hist",
        bins=100,
        title=title4,
        color=color2,
        ax=ax4,
    )
    plt.show()


def distplot_viz(
    data,
    column,
    separate_column,
    condition_1,
    condition_2,
    label1,
    label2,
    title,
    color1=None,
    color2=None,
):
    """
    Gets a Python Pandas dataframe and visualize displot by a column's conditions. It shows density of column.
    :param data: Dataframe to be analyze
    :param column: This column is for showing data distribution.
    :param separate_column: this colum is for creating histogram by a column's conditions.
    :param condition_1: It designates condition of separate column.
    :param condition_2: It designates condition of separate column.
    :param label1: It designates label by condition_1.
    :param label2: It designates label by condition_2.
    :param title: It designates title for graph.
    :param color1: It designates color for condition_1.
    :param color2: It designates color for condition_2.
    :return: This function doesn't return anything.

    """

    plt.figure(figsize=(15, 5))
    sns.distplot(
        data[data[separate_column] == condition_1][column],
        hist=False,
        kde=True,
        kde_kws={"shade": True, "linewidth": 3},
        label=label1,
    )
    sns.distplot(
        data[data[separate_column] == condition_2][column],
        hist=False,
        kde=True,
        kde_kws={"shade": True, "linewidth": 3},
        label=label2,
    )
    plt.title(title, fontsize=17)
    plt.xlabel(column, fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.legend()

def distplot_log_viz(
    data,
    column,
    title1,
    title2,
    color1=None,
    color2=None,
):
    """
    Gets a Python Pandas dataframe and visualize two distplot. One of they is by a column's conditions and other one of they is by gets its log. It shows density of column.
    :param data: Dataframe to be analyze
    :param column: This column is for showing data distribution.
    :param title1: It designates title for graph1.
    :param title2: It designates title for graph2.
    :param color1: It designates color for condition_1.
    :param color2: It designates color for condition_2.
    :return: This function doesn't return anything.

    """
    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    time_val = data[column].values

    sns.distplot(time_val, ax=ax[0], color=color1)
    ax[0].set_title(title1, fontsize=14)
    ax[1].set_xlim([min(time_val), max(time_val)])

    sns.distplot(np.log(time_val), ax=ax[1], color=color2)
    ax[1].set_title(title2, fontsize=14)
    ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

    plt.show()      
    
    
    
def correlation_chart(df, column_list):
    """
    This function gets a Python Pandas dataframe and list of columns to visualize correlations among those columns.
    :param df: Dataframe to be analyze
    :param column_list: List of columns to filter out only numeric columns to use in the fuction"
    :return: This function doesn't return anything.
    """
    plt.figure(figsize=(12, 8))
    plt.title("Correlation of Features for Data Frame")
    sns.heatmap(df[column_list].corr(), vmax=1.0, annot=True, cmap="mako")
    plt.show()
