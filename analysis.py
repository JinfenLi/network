"""
    author: Jinfen Li
    GitHub: https://github.com/LiJinfen
"""


import collections
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from scipy.stats import wasserstein_distance


def plot_distribution(dis, xlabel, title, up):
    kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 5})
    plt.figure(figsize=(10, 7), dpi=100)
    color = ['khaki', 'gold', 'sandybrown', 'orange', 'darkorange', 'tomato','peru', 'sienna', 'brown', 'black']
    for year, value in dis.items():
        sns.distplot(value, color=color.pop(0), label=str(year), **kwargs)

    plt.xlim(0,up)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel('frequency',fontsize=20)
    plt.title(title,fontsize=20)
    plt.legend()
    plt.show()


def plot_line(dis, dis2, ylabel, title, label1 = 'left', label2 = 'new'):
    x = list(dis.keys())
    y1 = list(dis.values())
    y2 = list(dis2.values())
    fig, ax = plt.subplots()
    ax.plot(x, y1, marker='*', label=label1)
    ax.plot(x, y2, marker='*', label=label2)
    ax.set_xlabel('year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.show()


def number_of_director(total_summary_info, start, end):
    number_of_director = collections.defaultdict(list)
    for year in range(start+1, end+1):
        new_boards = set(total_summary_info[year].keys()) - set(total_summary_info[year-1].keys())
        for bid in new_boards:
            number_of_director[year].append(len(total_summary_info[year][bid].keys()))

    plot_distribution(number_of_director, 'number of directors', 'distribution of number of directors in new boards', 18)


def director_in_board(total_summary_info, start, end):

    left_rate = collections.defaultdict(list)
    new_rate = collections.defaultdict(list)
    for year in range(start+1, end+1):
        remain_bids = set(total_summary_info[year-1].keys()) & set(total_summary_info[year].keys())
        for fbid in remain_bids:
            base = len(total_summary_info[year-1][fbid])
            left = set(total_summary_info[year-1][fbid].keys())-set(total_summary_info[year][fbid].keys())
            left_rate[year].append(len(left)/base)
            new = set(total_summary_info[year][fbid].keys()) - set(total_summary_info[year-1][fbid].keys())
            new_rate[year].append(len(new) / base)

    plot_distribution(left_rate, '# left directors/# total directors in previous year', 'distribution of percentages of left directors',1)
    plot_distribution(new_rate, '# new directors/# total directors in previous year', 'distribution of percentages of new directors',1)


def board(total_summary_info, start, end):
    left_board = {}
    new_board = {}
    for year in range(start+1, end+1):
        base = set(total_summary_info[year-1].keys())
        new_board[year] = len(set(total_summary_info[year].keys()) -base)/len(base)
        left_board[year] = len(base-set(total_summary_info[year].keys()))/len(base)

    plot_line(left_board, new_board, 'percentage', 'percentage of disband and new boards')


def director_in_network(total_summary_info, start, end):
    left_director = {}
    new_director = {}
    for year in range(start+1, end+1):
        base =set(np.hstack([ list(k.keys()) for k in total_summary_info[year - 1].values()]))
        new_director[year] = len(set(np.hstack([ list(k.keys()) for k in total_summary_info[year].values()])) - base) / len(base)
        left_director[year] = len(base -set(np.hstack([list(k.keys()) for k in total_summary_info[year].values()]))) / len(base)

    plot_line(left_director, new_director, 'percentage', 'percentage of left and new directors')


def gender_dis_annual(total_summary_info):
    dis = []
    males = {}
    females = {}
    for year, infos in total_summary_info.items():
        for bid, info in infos.items():
            dis.extend(info.values())
        counter = collections.Counter(dis)
        males[year] = counter['M']/sum(counter.values())
        females[year] = counter['F']/sum(counter.values())

    plot_line(males, females, 'percentage', 'gender distribution in network level', 'male', 'female')


def disbanded_board(total_summary_info, start, end):
    dis = collections.defaultdict(list)
    for year in range(start+1, end+1):
        base = set(total_summary_info[year-1].keys())
        disband_bids = base - set(total_summary_info[year].keys())
        for bid in disband_bids:
            dis[year].append(len(total_summary_info[year-1][bid]))

    plot_distribution(dis, "number of directors", "number of directors of disbanded boards", 12)


def remaining_board(total_summary_info, start, end):
    dis = collections.defaultdict(list)
    all_new_dids = {}
    all_old_dids = {}

    for year in range(start+1, end+1):
        base =set(np.hstack([list(k.keys()) for k in total_summary_info[year - 1].values()]))
        all_new_dids[year] = set(np.hstack([list(k.keys()) for k in total_summary_info[year].values()])) - base
        all_old_dids[year] = base & set(np.hstack([list(k.keys()) for k in total_summary_info[year].values()]))

    for year in range(start + 1, end + 1):
        base = set(total_summary_info[year - 1].keys())
        remain_bids = base & set(total_summary_info[year].keys())

        for bid in remain_bids:
            new_dids = set(total_summary_info[year][bid].keys()) - set(total_summary_info[year-1][bid].keys())
            if new_dids:
                dis[year].append(len(new_dids&all_new_dids[year])/len(new_dids))

    plot_distribution(dis, "percentage of new directors", "percentage of new directors in new connections in remaining boards", 1)


def new_board(total_summary_info, start, end):
    dis = collections.defaultdict(list)
    all_new_dids = {}
    all_old_dids = {}

    for year in range(start+1, end+1):
        base =set(np.hstack([ list(k.keys()) for k in total_summary_info[year - 1].values()]))
        all_new_dids[year] = set(np.hstack([ list(k.keys()) for k in total_summary_info[year].values()])) - base
        all_old_dids[year] = base & set(np.hstack([ list(k.keys()) for k in total_summary_info[year].values()]))

    for year in range(start + 1, end + 1):
        base = set(total_summary_info[year - 1].keys())
        new_bids = set(total_summary_info[year].keys()) - base
        for bid in new_bids:
            new_dids = set(total_summary_info[year][bid].keys())
            dis[year].append(len(new_dids&all_old_dids[year])/len(new_dids))

    plot_distribution(dis, "percentage of old directors", "percentage of old directors in new boards", 1)


def gender_law(total_summary_info, start, end):
    dis = collections.defaultdict(list)
    for year, infos in total_summary_info.items():
        for bid, info in infos.items():
            counter = collections.Counter(info.values())
            dis[year].append(counter['F']/sum(counter.values()))

    plot_distribution(dis, "percentage of females", "proportion of females in board level", 1)


def cal_mean_var(dis):
    return np.round(np.mean(dis), 2), np.round(np.var(dis), 2)


def plot_degree_distribution(degree_distributions, initial_distribution, prob,year):
    print("male:female = %d:%d" % (len(initial_distribution['M']), len(initial_distribution['F'])))

    def axis(dis):

        c = collections.Counter(dis)
        counter = dict(sorted(c.items(), key=lambda i: i[0]))
        x = list(counter.keys())
        y = [v / sum(list(counter.values())) for v in counter.values()]
        print(sum(list(counter.values())))
        return x, y

    male_x1, male_y1 = axis(degree_distributions['M'])
    female_x1, female_y1 = axis(degree_distributions['F'])
    male_x0, male_y0 = axis(initial_distribution['M'])
    female_x0, female_y0 = axis(initial_distribution['F'])

    mean_male_1, var_male_1 = cal_mean_var(degree_distributions['M'])
    mean_female_1, var_female_1 = cal_mean_var(degree_distributions['F'])
    mean_male_0, var_male_0 = cal_mean_var(initial_distribution['M'])
    mean_female_0, var_female_0 = cal_mean_var(initial_distribution['F'])

    fig, ax = plt.subplots()
    ax.plot(male_x1, male_y1, marker='*', label='synthetic male'+', var:'+str(var_male_1) +', mean:'+str(mean_male_1), color='midnightblue')
    ax.plot(female_x1, female_y1, marker='*', label='synthetic female'+', var:'+str(var_female_1) +', mean:'+str(mean_female_1), color='brown')
    ax.plot(male_x0, male_y0, marker='*', label='original male'+', var:'+str(var_male_0) +', mean:'+str(mean_male_0), color='dodgerblue')
    ax.plot(female_x0, female_y0, marker='*', label='original female'+', var:'+str(var_female_0) +', mean:'+str(mean_female_0), color='red')
    ax.set_xlabel('degree')
    ax.set_ylabel('P(degree)')
    if prob == 0:
        ax.set_title('degree distribution in %d degree based graph' % (year))
    elif prob == 1:
        ax.set_title('degree distribution in %d gender diversity based graph' % (year))
    ax.legend(loc=0)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.tight_layout()
    plt.show()


def total_degree(total_summary_info, did_year):
    total_gender_degree = {}
    for year, infos in total_summary_info.items():
        gender_d = collections.defaultdict(dict)
        gender_d['M'] = {}
        gender_d['F'] = {}
        for did, gender in did_year[year].items():
            gender_d[gender][did] = 0
        for bid, info in infos.items():
            for did, gender in info.items():
                gender_d[gender][did]+=len(info)-1
                # gender_d[gender][did] += 1
        gender_dis = collections.defaultdict(list)
        for gender, infos in gender_d.items():
            gender_dis[gender].extend(infos.values())
        total_gender_degree[year] = gender_dis
    return total_gender_degree


def pad_0_degree(summary):
    did_year = {}
    for year in summary.keys():
        if year-1 in did_year:
            did_year[year] = copy.deepcopy(did_year[year-1])
        else:
            did_year[year] = {}

        for bid, info in summary[year].items():
            for did, gender in info.items():
                did_year[year][did] = gender
    return did_year


def degree_distribution(synthetic_summary, original_summary, year):


    did_year_0 = pad_0_degree(synthetic_summary[0])
    original_did_year = pad_0_degree(original_summary)
    total_gender_degree0 = total_degree(synthetic_summary[0], did_year_0)

    original_gender_degree = total_degree(original_summary, original_did_year)

    plot_degree_distribution(total_gender_degree0[year], original_gender_degree[year], prob=0, year=year)
    if 10 in synthetic_summary:
        did_year_1 = pad_0_degree(synthetic_summary[10])
        total_gender_degree1 = total_degree(synthetic_summary[10], did_year_1)
        plot_degree_distribution(total_gender_degree1[year], original_gender_degree[year], prob=1, year=year)


# EMD between real and synthetic networks
def distance(synthetic_summary, original_summary, year, start):
    # {p:{year:{gender:[degree1, degree2,...]}}}

    def axis(dis):

        c = collections.Counter(dis)
        counter = dict(sorted(c.items(), key=lambda i: i[0]))
        p = {k:v/sum(counter.values()) for k,v in counter.items()}
        return p

    did_year_0 = pad_0_degree(original_summary)
    total_gender_degree0 = total_degree(original_summary, did_year_0)

    pf0 = axis(total_gender_degree0[start]['F'])
    pm0 = axis(total_gender_degree0[start]['M'])

    x = ['initial']
    y_f0, y_m0, y_cross0,y_cross_original = [], [], [], []

    output = wasserstein_distance(list(pf0.keys()), list(pm0.keys()), list(pf0.values()), list(pm0.values()))
    y_cross0.append(output)
    y_cross_original.append(output)

    for p, infos in synthetic_summary.items():
        x.append(round(p*0.1,1))
        did_year = pad_0_degree(synthetic_summary[p])
        total_gender_degree = total_degree(infos, did_year)
        info = total_gender_degree[year]

        pf1 = axis(info['F'])
        pm1 = axis(info['M'])

        output = wasserstein_distance(list(pm0.keys()), list(pm1.keys()), list(pm0.values()), list(pm1.values()))
        y_m0.append(output)

        output = wasserstein_distance(list(pf0.keys()), list(pf1.keys()), list(pf0.values()), list(pf1.values()))
        y_f0.append(output)

        output = wasserstein_distance(list(pm1.keys()), list(pf1.keys()), list(pm1.values()), list(pf1.values()))
        y_cross0.append(output)

    fig, ax = plt.subplots()
    ax.plot(x, y_cross0, marker='*')
    ax.set_xlabel('probability of an edge based on diversity')
    ax.set_ylabel('EMD')
    ax.set_title('EMD between distributions of male and female in %d'%year)
    ax.legend(loc=0)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x[1:], y_m0, marker='*', label='male')
    ax.plot(x[1:], y_f0, marker='*', label='female')
    ax.set_xlabel('probability of an edge based on diversity')
    ax.set_ylabel('EMD')
    ax.set_title('EMD between distributions of new graph \n and old graph in different genders in %d'%year)
    ax.legend(loc=0)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


# EMD between two iterations
def distance2(synthetic_summary, synthetic_summary2, year, start):
    # {p:{year:{gender:[degree1, degree2,...]}}}

    def axis(dis):

        c = collections.Counter(dis)
        counter = dict(sorted(c.items(), key=lambda i: i[0]))
        p = {k:v/sum(counter.values()) for k,v in counter.items()}
        return p

    x = ['initial']
    y_f0, y_m0 = [], []
    for p in range(11):

        x.append(round(p*0.1,1))
        did_year = pad_0_degree(synthetic_summary[p])
        total_gender_degree = total_degree(synthetic_summary[p], did_year)
        info = total_gender_degree[year]
        pf0 = axis(info['F'])
        pm0 = axis(info['M'])

        did_year = pad_0_degree(synthetic_summary2[p])
        total_gender_degree = total_degree(synthetic_summary2[p], did_year)
        info = total_gender_degree[year]
        pf1 = axis(info['F'])
        pm1 = axis(info['M'])

        output = wasserstein_distance(list(pm0.keys()), list(pm1.keys()), list(pm0.values()), list(pm1.values()))
        y_m0.append(output)

        output = wasserstein_distance(list(pf0.keys()), list(pf1.keys()), list(pf0.values()), list(pf1.values()))
        y_f0.append(output)

    fig, ax = plt.subplots()
    ax.plot(x[1:], y_m0, marker='*', label='male')
    ax.plot(x[1:], y_f0, marker='*', label='female')
    ax.set_xlabel('probability of an edge based on diversity')
    ax.set_ylabel('EMD')
    ax.set_title('EMD between distributions of graphs in two iterations in %d'%year)
    ax.legend(loc=0)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    with open('data/wrd/summary.pkl', 'rb') as file:
        total_summary_info = pickle.load(file)
    director_in_board(total_summary_info, 2005, 2014)
    board(total_summary_info, 2005, 2014)
    director_in_network(total_summary_info, 2005, 2014)
    number_of_director(total_summary_info, 2005, 2014)
    gender_dis_annual(total_summary_info)
    disbanded_board(total_summary_info, 2005, 2014)
    remaining_board(total_summary_info, 2005, 2014)
    new_board(total_summary_info, 2005, 2014)
    gender_law(total_summary_info, 2005, 2014)