
# Author: Zdawang
# Date: 2017-12-13
# Describe: 使用HMM模型进行中文分词的实现。


import pandas as pd
import numpy as np
from collections import defaultdict
from functools import reduce

class HMM(object):
    """
    HMM模型
    """
    def __init__(self, status, outputs):
        self.status = {s: i for i, s in enumerate(status)}
        self.outputs = {o: i for i, o in enumerate(outputs)}

        self.status_dict = {i: s for i, s in enumerate(status)}

        self.status_num = len(status) #状态个数
        self.outputs_num = len(outputs) #输出个数

        self.init_prob = [1/self.status_num] * self.status_num # 初始状态概率
        self.status_matrix = [[1/self.status_num] * self.status_num for _ in range(self.status_num)] #状态转移矩阵
        self.out_matrix = [[1/self.outputs_num] * self.outputs_num for _ in range(self.status_num)] #输出概率矩阵

    #计算概率，没使用EM算法，所以没用到
    def forward(self, outputs):
        """
        前向算法，给定输出O与模型，计算输出O的概率
        """
        if not outputs: return
        T = len(outputs)
        #将output转换为下标
        out = [self.outputs[o] for o in outputs]
        dp = [[0] * self.status_num for _ in range(T)]
        #初始化
        for i in range(self.status_num):
            dp[0][i] = self.init_prob[i] * self.out_matrix[i][out[0]]
        #迭代
        for t in range(1, T):
            for i in range(self.status_num):
                dp[t][i] = sum((dp[t-1][j] * self.status_matrix[j][i] for j in range(self.status_num))) * self.out_matrix[i][out[t]]
        return dp

    def backward(self, outputs):
        """
        后向算法，给定输出O与模型，计算输出O的概率
        """
        if not outputs: return
        T = len(outputs)
        #将output转换为下标
        out = [self.outputs[o] for o in outputs]
        dp = [[0] * self.status_num for _ in range(T)]
        #初始化
        for i in range(self.status_num):
            dp[T-1][i] = 1
        #迭代
        for t in range(T-2, -1, -1):
            for i in range(self.status_num):
                dp[t][i] = sum(self.status_matrix[i][j] * self.out_matrix[j][out[t+1]] * dp[t+1][j] for j in range(self.status_num))
        return dp

    #学习算法,最大似然算法
    def maximum_likehood(self, words, tags):
        #统计单个句子
        def count(word, tag):
            if not tag: return
            #转为数字下标
            tag = [self.status[t] for t in tag]
            word = [self.outputs[w] for w in word]
            #统计
            init[tag[0]] += 1
            out_tran[tag[0]][word[0]] += 1
            for i in range(1, len(tag)):
                status_tran[tag[i-1]][tag[i]] += 1
                out_tran[tag[i]][word[i]] += 1

        init = [0] * self.status_num # 初始状态概率
        status_tran = [[0] * self.status_num for _ in range(self.status_num)] #状态转移矩阵
        out_tran = [[0] * self.outputs_num for _ in range(self.status_num)] #输出概率矩阵

        list(map(count, words, tags))
        #更新init_prob
        init_sum = sum(init)
        for i in range(self.status_num):
            self.init_prob[i] = init[i]/init_sum
        #status_matrix
        for i in range(self.status_num):
            status_sum = sum(status_tran[i])
            for j in range(self.status_num):
                self.status_matrix[i][j] = status_tran[i][j]/status_sum
        #out_matrix
        for i in range(self.status_num):
            out_sum = sum(out_tran[i])
            for j in range(self.outputs_num):
                self.out_matrix[i][j] = out_tran[i][j]/out_sum


    #预测算法
    def predict(self, words):
        if not words: return
        T = len(words)
        #将output转换为下标
        out = [self.outputs[o] for o in words]
        dp = [[0] * self.status_num for _ in range(T)]
        path = [[0] * self.status_num for _ in range(T)]
        #初始化
        for i in range(self.status_num):
            dp[0][i] = self.init_prob[i] * self.out_matrix[i][out[0]]
            path[0][i] = 0
        #迭代
        for t in range(1, T):
            for i in range(self.status_num):
                tmp = [dp[t-1][j] * self.status_matrix[j][i] for j in range(self.status_num)]
                dp[t][i] = max(tmp) * self.out_matrix[i][out[t]]
                path[t][i] = tmp.index(max(tmp))
        #得到最优解
        res_path = [0] * T
        P = max(dp[T-1])
        res_path[T-1] = dp[T-1].index(P)
        #路径回溯
        for t in range(T - 2, -1, -1):
            res_path[t] = path[t+1][res_path[t+1]]
        return "".join([self.status_dict[s] for s in res_path])


#将训练数据转为BEMS的形式
def encode(words):
    words_list = words.split(" ")
    res = []
    for word in words_list:
        l = len(word)
        if l == 0:
            continue
        if l == 1:
            res.append("S")
        else:
            res.append("B" + (l-2)*"M" + "E")
    return "".join(res)

#将BEMS转为结果
def decode(words_tags):
    words, tags = words_tags[0], words_tags[1]

    res, start = [], 0
    for i, s in enumerate(tags):
        if s == "S":
            res.append(words[i])
        elif s == "B":
            start = i
        elif s == "M":
            continue
        else:
            res.append(words[start: i + 1])
    return "  ".join(res)

def get_num(x):
    ans_set, res_set = set(x[0].split("  ")), set(x[1].split("  "))
    get_num.answer_num += len(ans_set)
    get_num.result_num += len(res_set)
    get_num.result_correct_num += len(ans_set & res_set)

#************************

train_path = "icwb2-data/training/msr_training.txt"
dev_path = "icwb2-data/testing/msr_test.txt"
dev_label_path = "icwb2-data/gold/msr_test_gold.txt" 

train_data = pd.read_csv(train_path, encoding = "gbk", names = ["correct_answer"])
dev_data = pd.read_csv(dev_path, encoding = "gbk", names = ["origin"])
dev_label = pd.read_csv(dev_label_path, encoding = "gbk", names = ["correct_answer"])
dev_data = pd.concat([dev_data, dev_label], axis = 1)


#正确结果，去掉首尾空格
train_data["correct_answer"] = train_data["correct_answer"].map(lambda x: x.strip())
#未切分数据
train_data["origin"] = train_data["correct_answer"].map(lambda x: "".join(x.split(" ")))
#标注
train_data["correct_tags"] = train_data["correct_answer"].map(encode)


#构建hmm模型
outputs = reduce(lambda s, x: s | set(x), list(train_data["origin"]), set())
outputs = reduce(lambda s, x: s | set(x), list(dev_data["origin"]), outputs)

status = ["B", "E", "M", "S"]
hmm = HMM(status, outputs)
#参数学习
hmm.maximum_likehood(list(train_data["origin"]), list(train_data["correct_tags"]))


#预测
dev_data["hmm_tags"] = dev_data["origin"].map(hmm.predict)
#切分结果
dev_data["result"] = dev_data[["origin", "hmm_tags"]].apply(decode, axis = 1)


#计算P,R,F
get_num.answer_num, get_num.result_num, get_num.result_correct_num = 0, 0, 0
dev_data[["correct_answer", "result"]].apply(get_num, axis = 1)

P = get_num.result_correct_num/get_num.answer_num
R = get_num.result_correct_num/get_num.result_num

F1 = 2*P*R/(P + R)
print("P is: ", P)
print("R is: ", R)
print("F1 is: ", F1)