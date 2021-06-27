import matplotlib.pyplot as plt

def histogram(data, bins=20, xlabel = 'data', ylabel='counts'):
    '''
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    '''
    plt.hist(data, bins=bins, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()