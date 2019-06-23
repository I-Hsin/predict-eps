import csv
import os
import sys

if __name__ == '__main__':
    path = os.getcwd() + "//data//"
    lst = os.listdir(path)
    # print lst
    # item = lst[0]
    # print "item", item
    # # item = "000001_xx.csv"
    # src = path + item
    # dst = path + item[:item.find('_')] + ".csv"
    # print os.getcwd()
    # print src
    # print dst
    #
    # os.rename(src, dst)
    # sys.exit(0)
    # for item in lst:
    #     src = path + item
    #     dst = path + item[:item.find('_')] + ".csv"
    #     print src
    #     print dst
    #     os.rename(src, dst)


    f = open(path + lst[0], 'r')
    for row in csv.reader(f):
        print row
    f.close()
