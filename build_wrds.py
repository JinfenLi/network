import networkx as nx
import collections
import numpy as np
import copy
import pickle
import argparse
from scipy.stats import truncnorm
import analysis
global initial



class Graph:
    def __init__(self, p_edge, director_pool, max_n_serving, max_n_director, new_board_rate, director_to_pool_rate, disband_rate, disband_p):
        self.p_edge = p_edge
        self.director_pool = director_pool
        self.max_n_serving = max_n_serving
        self.max_n_director = max_n_director
        self.new_board_rate = new_board_rate
        self.director_to_pool_rate = director_to_pool_rate
        self.disband_rate = disband_rate
        self.gender_info = collections.defaultdict(int)
        self.graph = nx.Graph()
        self.disband_p = disband_p


    def assign_gender(self):
        # gender_counter = collections.Counter([v['gender'] for k,v in self.director_pool.items()])
        male_ratio = self.gender_info['M']/ (self.gender_info['M']+self.gender_info['F'])
        gender = np.random.choice(['M', 'F'], 1, p=[male_ratio, 1-male_ratio])[0]
        return gender


    def cal_wei(self, links1, links2):
        # calculate the desire level of male and female, aim to make link2 close to link1
        s1 = sum(links1) + 1
        s2 = sum(links2) + 1
        w_links1 = [l / s1 for l in links1]
        w_links2 = [l / s2 for l in links2]
        w = []
        for i in range(len(links1)):
            w.append(np.exp(w_links1[i] - w_links2[i]))
        return w


    def diversity_choice(self, d_nei, candidates):


        counter = collections.Counter(d_nei)
        weights = self.cal_wei([self.gender_info['M'], self.gender_info['F']],
                                    [counter['M'], counter['F']])
        weights = [w / sum(weights) for w in weights]
        d_target = np.random.choice(['M', 'F'], 1, p=weights)[0]
        gender_candidates = [d for d in candidates if self.director_pool[d] == d_target]
        degree_weights = [self.graph.degree[d] if d in self.graph.nodes else 0 for d in gender_candidates]
        if not gender_candidates:
            gender_candidates = [d for d in candidates if self.director_pool[d] == 'M']
            degree_weights = [self.graph.degree[d] if d in self.graph.nodes else 0 for d in gender_candidates]
        if sum(degree_weights) == 0:
            norm_node_weights = [1 / len(degree_weights)] * len(degree_weights)
        else:
            norm_node_weights = [w / sum(degree_weights) for w in degree_weights]
        node_target = np.random.choice(gender_candidates, 1, p=norm_node_weights)[0]

        return node_target


    def add_synthetic_directors(self, bid, n_director):
        candidates = copy.deepcopy(self.legal_candidates())
        d_nei = []
        for edge in range(int(n_director)):

            edge_type = np.random.choice(['diversity', 'degree'], 1, p=[self.p_edge, 1 - self.p_edge])[0]

            if edge_type == 'degree':
                weights = [self.graph.degree[d] if d in self.graph.nodes else 0 for d in candidates]
                if sum(weights) != 0:
                    norm_weight = [w / sum(weights) for w in weights]
                else:
                    norm_weight = [1 / len(weights) for w in weights]
                cur_target = np.random.choice(candidates, 1, p=norm_weight)[0]

            else:
                cur_target = self.diversity_choice(d_nei, [d for d in candidates])

            d_target = self.director_pool[cur_target]
            candidates.remove(cur_target)
            d_nei.append(d_target)

            self.graph.add_edges_from([(bid, cur_target)])
            self.graph.add_nodes_from([cur_target], bipartite=0)


    def add_real_directors(self, bid, n_director):
        candidates = copy.deepcopy(self.legal_candidates())
        new_candidates = [c for c in candidates if c not in self.graph.nodes]
        old_candidates = list(set(candidates) - set(new_candidates))
        d_nei = []
        for edge in range(int(n_director)):
            # new directors have higher probability than old directors to join in the new board
            edge_type = np.random.choice(['n', 'o'], 1, p=[0.9, 0.1])[0]

            if edge_type == 'n':

                cur_target = np.random.choice(new_candidates, 1)[0]
                new_candidates.remove(cur_target)


            else:
                weights = [1/self.graph.degree[d] for d in old_candidates]
                if sum(weights) != 0:
                    norm_weight = [w / sum(weights) for w in weights]
                else:
                    norm_weight = [1 / len(weights) for w in weights]
                cur_target = np.random.choice(old_candidates, 1, p=norm_weight)[0]
                old_candidates.remove(cur_target)

            d_target = self.director_pool[cur_target]
            # candidates.remove(cur_target)
            d_nei.append(d_target)

            self.graph.add_edges_from([(bid, cur_target)])
            self.graph.add_nodes_from([cur_target], bipartite=0)



    def legal_candidates(self, remove=[]):

        return [k for k,v in self.director_pool.items() if k not in remove and ((k in self.graph.nodes and self.graph.degree[k]<=5) or k not in self.graph.nodes)]



    def initial_graph(self, path, start):
        graph = nx.Graph()
        with open(path, 'rb') as file:
            total_summary_info = pickle.load(file)
        bids = set()
        dids = set()
        gender_d = {}
        for bid, v in total_summary_info[start].items():
            bids.add(bid)
            for did, gender in v.items():
                dids.add(did)
                graph.add_edges_from([(str(bid)+'b', str(did)+'a')])
                gender_d[did] = gender
        # reassign node label
        bids = sorted(list(bids))
        dids = sorted(list(dids))
        graph_d = {}
        for i, bid in enumerate(bids):
            graph_d[str(bid)+'b'] = str(i) +'b'
        for i, did in enumerate(dids):
            graph_d[str(did)+'a'] = str(i) +'a'
            self.director_pool[str(i) +'a'] = gender_d[did]
        self.graph =nx.relabel_nodes(graph, graph_d)
        self.gender_info['M'] = collections.Counter(gender_d.values())['M']
        self.gender_info['F'] = collections.Counter(gender_d.values())['F']



    def build_graph(self, board_id, type):


        old_boards, old_directors = [node for node in self.graph.nodes if 'b' in node], \
                                    [node for node in self.graph.nodes if 'a' in node]
        
        # add new directors in the director pool
        for _ in range(int(len(old_directors)*self.director_to_pool_rate)):
            gender = self.assign_gender()
            self.gender_info[gender] += 1
            self.director_pool[str(len(self.director_pool)) + 'a'] = gender
        
        # disband old boards using the probability dictionary in disband_p()
        previous_board_num = len(old_boards)
        weights = [self.disband_p[self.graph.degree[bid]] for bid in old_boards]
        norm_weights = [w/sum(weights) for w in weights]
        disband_boards = np.random.choice(old_boards, int(previous_board_num * self.disband_rate),p=norm_weights, replace=False)
        for bid in disband_boards:
            self.graph.remove_node(bid)
        
        # deal with remaining boards
        remain_boards = set(old_boards) - set(disband_boards)
        for bid in remain_boards:

            original_director = copy.deepcopy(list(self.graph.adj[bid]))
            # remove directors
            if type == 'wrd':
                remove_directors_ratio = get_truncated_normal()
            '''
            to do:
            elif type == 'bag':
                remove_directors_ratio = get_truncated_normal(mean=0, sd=0.01, low=0, upp=1.0)
            '''

            remove_ids = list(np.random.choice(original_director, int(len(original_director) * remove_directors_ratio), replace=False))
            while remove_ids:
                did = remove_ids.pop()
                self.graph.remove_edge(bid, did)
            
            # add directors
            cur_num_director = len(list(self.graph.adj[bid]))
            if type == 'wrd':
                director_to_oldboard_rate = get_truncated_normal()
            '''
            to do:
            elif type == 'bag':
                remove_directors_ratio = get_truncated_normal(mean=0, sd=0.01, low=0, upp=1.0)
            '''
            add_candidates = self.legal_candidates(original_director)
            # old directors have probability of 0.1 and new directors have probability of 0.9 joining in the ramining boards
            join_p = [0.1 if ac in self.graph.nodes else 0.9 for ac in add_candidates]
            norm_add_p = [p/sum(join_p) for p in join_p]
            # min: cannot exceed max_n_serving_board
            add_ids = list(np.random.choice(add_candidates, min(int(len(original_director) * director_to_oldboard_rate), len(original_director) - cur_num_director),
                                           p=norm_add_p, replace=False))
            
            while add_ids:
                did = add_ids.pop()
                self.graph.add_edges_from([(bid, did)])
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
        # create new boards
        for _ in range(int(previous_board_num * (self.new_board_rate))):



            
            if type == 'wrd':
                n_director = get_truncated_normal(mean=6, sd=2, low=2, upp=18)
            elif type == 'bag':
                n_director = get_truncated_normal(mean=3, sd=1, low=2, upp=18)

            if args.is_simulate_real:
                self.add_real_directors(str(board_id) + 'b', n_director)
            else:
                self.add_synthetic_directors(str(board_id) + 'b', n_director)
            self.graph.add_nodes_from([str(board_id)+'b'], bipartite=1)
            board_id += 1
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

        return board_id


def disband_p():
    p = [9.40708913, 8.3845761, 7.32838017, 7.04114693, 6.96113275, 6.89015389, 6.74661214, 6.74109047, 6.33027024,
     6.28361533, 6.00508119, 5.29230988]
    pp = [pp/sum(p) for pp in p]
    d = {}
    # depends on number of directors in that board
    for num in [7, 6, 8, 5, 4, 3, 9, 2, 10, 1, 11, 12]:
        d[num] = pp.pop(0)
    for num in [1,13,14,15,16,17,18]:
        d[num] = 0
    return d


def get_truncated_normal(mean=0, sd=0.2, low=0, upp=1.0):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(1)[0]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_analysis', type=bool, default=False)
    parser.add_argument('--is_simulate', type=bool, default=True)
    parser.add_argument('--is_simulate_real', type=bool, default=False)
    parser.add_argument('--path', type=str, default=r'wrd_graph', help='objects save path')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    type = args.path[:3]
    if type == 'wrd':
        start = 2005
        end = 2014


    if args.is_simulate:

        total_summary = collections.defaultdict(dict)

        # probability of an edge based on diversity: p_diversity range from 0 to 1
        for p_edge in range(11):
            if type == 'wrd':
                new_board_rate = [0.7, 0.5, 0.2, 0.05, 0.15, 0.2, 0.1, 0.2, 0.15]
                disband_rate = [0, 0.01, 0.15, 0.14, 0.05, 0.05, 0.05, 0.01, 0.25]
                director_to_pool_rate = [0.9, 0.8, 0.6, 0.45, 0.3, 0.5, 0.4, 0.25, 0.5]
            '''
            to do:
            elif type == 'bag':
                new_board_rate = [0.7, 0.5, 0.2, 0.05, 0.15, 0.2, 0.1, 0.2, 0.15]
                disband_rate = [0, 0.01, 0.15, 0.14, 0.05, 0.05, 0.05, 0.01, 0.25]
                director_to_pool_rate = [0.9, 0.8, 0.6, 0.45, 0.3, 0.5, 0.4, 0.25, 0.5]
            '''
            p_edge *= 0.1
            p_edge = round(p_edge, 2)
            G = Graph(p_edge=p_edge, director_pool={}, max_n_serving=6, max_n_director=16,
                      new_board_rate='', director_to_pool_rate='', disband_rate='', disband_p=disband_p())
            G.initial_graph("data/"+type +'/summary.pkl', start)
            old_boards = [node for node in G.graph.nodes if 'b' in node]
            summary = collections.defaultdict(dict)
            for bid in old_boards:
                for did in G.graph.adj[bid]:
                    summary[bid][did] = G.director_pool[did]
            total_summary[int(p_edge*10)][start] = summary
            board_id = len(old_boards)
            print("each edge has %s probability based on diversity" % p_edge)
            for year in range(start+1, end+1):

                G.new_board_rate = new_board_rate.pop(0)
                G.disband_rate = disband_rate.pop(0)
                G.director_to_pool_rate = director_to_pool_rate.pop(0)
                board_id = G.build_graph(board_id, type)
                print("generated %d boards" % board_id)
                summary = collections.defaultdict(dict)
                old_boards = [node for node in G.graph.nodes if 'b' in node]

                for bid in old_boards:
                    for did in G.graph.adj[bid]:
                        summary[bid][did] = G.director_pool[did]
                total_summary[int(p_edge*10)][year] = summary

            if args.is_analysis:
                if p_edge == 0 or p_edge==1:
                    analysis.director_in_board(total_summary[int(p_edge*10)], start, end)
                    analysis.number_of_director(total_summary[int(p_edge*10)], start, end)
                    analysis.board(total_summary[int(p_edge*10)], start, end)
                    analysis.director_in_network(total_summary[int(p_edge*10)], start, end)
            if args.is_simulate_real:
                break
        with open(args.path + '/'+ type + '.pkl', 'wb') as file:
            pickle.dump(total_summary, file)
    else:

        with open('data/'+type +'/summary.pkl', 'rb') as file:
            original_summary = pickle.load(file)
        with open(args.path + '/' + type + '.pkl', 'rb') as file:
            synthetic_summary0 = pickle.load(file)
        with open(args.path + '/'+type + '.pkl', 'rb') as file:
            synthetic_summary = pickle.load(file)
            for year in [2006, 2010, 2014]:
                analysis.degree_distribution(synthetic_summary, original_summary, year)
                # analysis.distance(synthetic_summary, original_summary, year, 2005)



