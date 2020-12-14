import os
import wrds
import collections
import pandas as pd
import pickle
import networkx as nx
import random

def dump_synthetic_data():

    def generate(n, p, y):
        # male count
        m1 = int(y * n)
        m2 = n - m1
        x = p
        p = p * 100
        G = nx.scale_free_graph(n)
        edges = list(set(list(G.edges())))
        G = nx.Graph()
        G.add_edges_from(edges)
        degrees = list(G.degree())
        degree_dic = {u: v for (u, v) in degrees}

        degree_dic = {k: v for k, v in sorted(degree_dic.items(), key=lambda item: item[1], reverse=True)}

        gender = {}

        n1 = 0
        n2 = 0

        for u in degree_dic.copy():
            if n1 < m1 and n2 < m2:
                r = random.randint(0, 101)
                if r < p:
                    gender[u] = 'male'
                    n1 += 1
                else:
                    gender[u] = 'female'
                    n2 += 1

            else:

                if n1 < m1:
                    gender[u] = 'male'
                    n1 += 1
                else:
                    gender[u] = 'female'
                    # n1 += 1

        nx.set_node_attributes(G, gender, 'gender')
        # print(nx.attribute_assortativity_coefficient(G, 'gender'))
        print(len(G.edges))

        nx.write_gpickle(G, 'data/DS/(' + str(n) + ')(' + str(x) + ')(' + str(y) + ')(1_1).gpickle')
        return G

    ns = [500]
    ps = [0.8]
    # male ratio
    xs = [0.99]
    for p in ps:
        for n in ns:
            for x in xs:
                G = generate(n, p, x)


def dump_boardex_data():


    db = wrds.Connection()
    company_networks = db.raw_sql("select * from jr_wrds_company_networks")
    profile = db.raw_sql("select * from jr_dir_profile_details")
    org_summary = db.raw_sql("select * from jr_wrds_org_summary")
    individual_networks = db.raw_sql("select * from jr_wrds_individual_networks")

    # filter individual only from the summary tables
    filter_dids = set(org_summary['directorid'].values)
    filter_bids = set(org_summary['boardid'].values)
    individual_networks = individual_networks[individual_networks['companyid'].isin(list(filter_bids))]
    individual_networks = individual_networks[individual_networks['dirbrdid'].isin(list(filter_dids))]
    individual_networks = individual_networks.replace('Curr', 2015)

    org_summary.to_excel('data/wrds/org_summary.xlsx', index=False)

    summary = pd.read_excel('data/wrds/org_summary.xlsx')
    total_summary_info = {}
    for year in range(2005, 2015):
        summary_info = collections.defaultdict(dict)
        sum_year = summary[summary['annualreportdate'].dt.year == year]
        for bid, groups in sum_year.groupby('boardid'):
            for did, gender in zip(groups['directorid'], groups['gender']):
                summary_info[bid][did] = gender
        total_summary_info[year] = summary_info

    with open('data/wrd/summary.pkl', 'wb') as file:
        pickle.dump(total_summary_info, file)


def dump_bag_networks():
    gender_d = {}
    with open('data_people.txt', 'r') as file:
        f = pd.read_csv(file, delimiter=' ')
        for i, gender in zip(f["id"].values, f["gender"].values):
            gender_d[i] = 'M' if gender == 1 else 'F'
    files = os.listdir('data/bag')
    datas = []
    summary = {}
    for file in files:
        with open(os.path.join('data/bag', file), 'r') as f:
            if file.endswith('txt'):
                year = file[:4]
                summar = collections.defaultdict(dict)
                lines = f.readlines()
                for line in lines:
                    bid, did = line.strip().split()
                    datas.append([year, bid, did, gender_d[int(did)]])
                    summar[bid][did] = gender_d[int(did)]
            if summar:
                summary[int(year)] = summar
    with open('data/bag/summary.pkl', 'wb') as file:
        pickle.dump(summary, file)


