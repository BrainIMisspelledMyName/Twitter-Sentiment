# -*- coding: utf-8 -*-

# Final Project
# CS175, Winter 2015
#
# Sentiment Analysis on Tweets 
#
# Authors:
#  Bing Hui Feng, 78912993, bhfeng@uci.edu 
#  Jessica Pham, 80859856, jessictp@uci.edu
#  Brian Chou, 72312823, bchou1@uci.edu

"""
    PROGRAM DESCRIPTION 
        used to take data and display it as a histogram 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.figure as fig
from matplotlib.font_manager import FontProperties

# make bargraph method for token frequency
def autobarlabeler(rects, values, ax):
    # attach some text labels
    for i in range(len(rects)):
        height = rects[i].get_height()
        ax.text(rects[i].get_x()+rects[i].get_width()/2, 1.02*height, '{0:.3g}'.format(values[i]),
                ha='center', va='bottom')



def bargraph(raw, ppd, third):    
    plot1 = plt             # i dont know why i did this probably for multiple plots or something
    N = len(raw)                   # number of bars per rectangle this one has 2
    width = 0.2             # the width of the bars
    ind = np.arange(N)      # the x locations for the groups
    
    fig, ax = plt.subplots() #does something im not sure
    
    rects1 = ax.bar(ind+.2, raw, width, color='b',alpha=.5)
    rects2 = ax.bar(ind+width+.2, ppd, width, color='g', alpha=.5)
    rects3 = ax.bar(ind+width+.4, ppd, width, color='r', alpha=.5)
    
    ax.set_ylabel('Scores')
    ax.set_title('cross-validation means of Classifers accuracies with no preprocessing')
    ax.set_xticks(ind+width+.3)
    ax.set_xticklabels( ('Unigrams', 'Uni & Bi', 'Bigrams') )
    
    
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('NB', 'LR','SVM'), prop = fontP,
                loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,.3,.9))

    autobarlabeler(rects1, raw, ax)
    autobarlabeler(rects2, ppd, ax)
    autobarlabeler(rects3, third, ax)
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8.5,4.5)
    fig.savefig('hist_nopre.png',dpi=100)
    plt.show()

def freq_histogram(alphab, frequencies):
    
    
    pos = np.arange(len(alphab))
    width = .3     # gives histogram aspect to the bar diagram
    
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(alphab)    
    ax.set_title('Frequency of words without stopword removal')
    
    plt.xticks(rotation='vertical')    
    plt.bar(pos, frequencies, width, color='b', alpha=.5)
    plt.show()
    
# keep all functions 
# add more or edit main    
if __name__ == "__main__":
    
    #currently doing no preprocessing
    # format Unigrams : 1  , bigrams : 2 , BOTH : 3 
    NB = [0.76489, 0.7838895, 0.727066] 
    LR = [0.786083, 0.7966055, 0.7298505] 
    svm = [0.778273 , 0.7956685, 0.7276935]
    
#    NB = [0.764885, 0.7835775, 0.747566] 
#    LR = [0.782313, 0.796348, 0.7426635] 
#    svm = [0.7746325 , 0.790304, 0.7389775]
    bargraph(NB, LR, svm)