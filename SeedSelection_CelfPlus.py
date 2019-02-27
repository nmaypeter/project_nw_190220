import operator

from Diffusion_NormalIC import *


class SeedSelectionNG:
    def __init__(self, g_dict, s_c_dict, prod_list, total_bud, monte):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### total_bud: (int) the budget to select seed
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### monte: (int) monte carlo times
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.total_budget = total_bud
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.monte = monte

    def getExpectedProfit(self, k_prod, i_node, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        s_set_t[k_prod].add(i_node)
        a_n_set = copy.deepcopy(s_set_t)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        # -- notice: prevent the node from owing no receiver --
        if i_node not in self.graph_dict:
            return round(ep, 4)

        # -- insert the children of seeds into try_s_n_sequence --
        ### try_s_n_sequence: (list) the sequence to store the seed for k-products [k, i]
        ### try_a_n_sequence: (list) the sequence to store the nodes may be activated for k-products [k, i, prob]
        try_s_n_sequence, try_a_n_sequence = [], []
        for k in range(self.num_product):
            for i in s_set_t[k]:
                try_s_n_sequence.append([k, i])

        while len(try_s_n_sequence) > 0:
            seed = choice(try_s_n_sequence)
            try_s_n_sequence.remove(seed)
            k_prod_t, i_node_t = seed[0], seed[1]

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, 1])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        while len(try_a_n_sequence) > 0:
            try_node = choice(try_a_n_sequence)
            try_a_n_sequence.remove(try_node)
            k_prod_t, i_node_t, child_depth = try_node[0], try_node[1], try_node[2]

            ### -- purchasing --
            ep += self.product_list[k_prod_t][0]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            if child_depth >= 3:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, child_depth + 1])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        a_n_set = copy.deepcopy(s_set_t)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        # -- insert the children of seeds into try_s_n_sequence --
        ### try_s_n_sequence: (list) the sequence to store the seed for k-products [k, i]
        ### try_a_n_sequence: (list) the sequence to store the nodes may be activated for k-products [k, i, prob]
        try_s_n_sequence, try_a_n_sequence = [], []
        for k in range(self.num_product):
            for i in s_set_t[k]:
                try_s_n_sequence.append([k, i])

        while len(try_s_n_sequence) > 0:
            seed = choice(try_s_n_sequence)
            try_s_n_sequence.remove(seed)
            k_prod_t, i_node_t = seed[0], seed[1]

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, 1])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        while len(try_a_n_sequence) > 0:
            try_node = choice(try_a_n_sequence)
            try_a_n_sequence.remove(try_node)
            k_prod_t, i_node_t, child_depth = try_node[0], try_node[1], try_node[2]

            ### -- purchasing --
            ep += self.product_list[k_prod_t][0]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            if child_depth >= 3:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, child_depth + 1])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)

    def generateCelfSequence(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) [k_prod, i_node, mg1, prev_best, mg2, flag]
        cur_best, cur_best_mg1 = [-1, '-1'], 0.0
        celf_seq = [[-1, '-1', 0.0, cur_best, 0.0, 0]]

        ssng_ss = SeedSelectionNG(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget, self.monte)

        for k in range(self.num_product):
            for i in set(self.graph_dict.keys()):
                # print(k, i)
                # -- the cost of seed cannot exceed the budget --
                if self.seed_cost_dict[i] > self.total_budget:
                    continue

                s_set = [set() for _ in range(self.num_product)]
                s_set[k].add(i)
                ep1 = 0.0
                for _ in range(self.monte):
                    ep1 += ssng_ss.getSeedSetProfit(s_set)
                ep1 = round(ep1 / self.monte, 4)
                mg1 = round(ep1, 4)
                mg2 = mg1

                if cur_best[1] != '-1':
                    s_set[cur_best[0]].add(cur_best[1])
                    ep2 = 0.0
                    for _ in range(self.monte):
                        ep2 += ssng_ss.getSeedSetProfit(s_set)
                    ep2 = round(ep2 / self.monte, 4)
                    mg2 = round(ep2, 4)

                celf_ep = [k, i, mg1, cur_best, mg2, 0]
                celf_seq.append(celf_ep)
                for celf_item in celf_seq:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                        celf_seq.pop()
                        break

                if celf_seq[0][2] > cur_best_mg1:
                    cur_best = [celf_seq[0][0], celf_seq[0][1]]
                    cur_best_mg1 = celf_seq[0][2]

        return celf_seq, cur_best


if __name__ == "__main__":
    data_set_name = "email_undirected"
    product_name = "r1p3n1"
    total_budget = 10
    pp_strategy = 1
    whether_passing_information_without_purchasing = bool(0)
    monte_carlo, eva_monte_carlo = 10, 100

    iniG = IniGraph(data_set_name)
    iniW = IniWallet(data_set_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()
    graph_dict = iniG.constructGraphDict()
    product_list = iniP.getProductList()
    wallet_list = iniW.getWalletList(product_name)
    num_node = len(seed_cost_dict)
    num_product = len(product_list)

    # -- initialization for each budget --
    start_time = time.time()

    ssng = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, total_budget, monte_carlo)
    eva = Evaluation(graph_dict, seed_cost_dict, product_list, pp_strategy, whether_passing_information_without_purchasing)

    personal_prob_list = eva.setPersonalProbList(wallet_list)

    ### result: (list) [profit, budget, seed number per product, customer number per product, seed set] in this execution_time
    result = []
    avg_profit, avg_budget = 0.0, 0.0
    avg_num_k_seed, avg_num_k_pn = [0 for _ in range(num_product)], [0 for _ in range(num_product)]
    profit_k_list, budget_k_list = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

    # -- initialization for each sample_number --
    ### now_profit, now_budget: (float) the profit and budget in this execution_time
    now_profit, now_budget = 0.0, 0.0
    ### seed_set: (list) the seed set
    seed_set = [set() for _ in range(num_product)]

    celf_sequence, cur_best_g = ssng.generateCelfSequence()
    mep_g = celf_sequence.pop(0)
    mep_k_prod, mep_i_node, mep_mg1, mep_prev_best, mep_mg2, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3], mep_g[4], mep_g[5]
    last_seed = [-1, '-1']

    while now_budget <= total_budget and mep_i_node != '-1':
        if now_budget + seed_cost_dict[mep_i_node] > total_budget:
            mep_g = celf_sequence.pop(0)
            mep_k_prod, mep_i_node, mep_mg1, mep_prev_best, mep_mg2, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3],  mep_g[4], mep_g[5]
            continue

        seed_set_length = 0
        for kk in range(num_product):
            seed_set_length += len(seed_set[kk])

        if mep_flag == seed_set_length:
            seed_set[mep_k_prod].add(mep_i_node)
            last_seed = [mep_k_prod, mep_i_node]

            budget_k_list[mep_k_prod] += seed_cost_dict[mep_i_node]
            now_budget += seed_cost_dict[mep_i_node]

            mep_g = celf_sequence.pop(0)
            mep_k_prod, mep_i_node, mep_mg1, mep_prev_best, mep_mg2, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3], mep_g[4], mep_g[5]
            continue

        elif operator.eq(mep_prev_best, last_seed):
            mep_mg1 = mep_mg2

        else:
            ep_g = 0.0
            for _ in range(monte_carlo):
                ep_g += ssng.getSeedSetProfit(seed_set)
            ep_g = round(ep_g / monte_carlo, 4)

            ep1_g = 0.0
            for _ in range(monte_carlo):
                ep1_g += ssng.getExpectedProfit(mep_k_prod, mep_i_node, seed_set)
            ep1_g = round(ep1_g / monte_carlo, 4)
            mep_mg1 = round(ep1_g - ep_g, 4)

            mep_prev_best = cur_best_g

            seed_set_t = copy.deepcopy(seed_set)
            seed_set_t[cur_best_g[0]].add(cur_best_g[1])
            ep_g = 0.0
            for _ in range(monte_carlo):
                ep_g += ssng.getSeedSetProfit(seed_set_t)
            ep_g = round(ep_g / monte_carlo, 4)
            ep2_g = 0.0
            for _ in range(monte_carlo):
                ep2_g += ssng.getExpectedProfit(mep_k_prod, mep_i_node, seed_set_t)
            ep2_g = round(ep2_g / monte_carlo, 4)
            mep_mg2 = round(ep2_g - ep_g, 4)

        seed_set_length = 0
        for kk in range(num_product):
            seed_set_length += len(seed_set[kk])
        mep_flag = seed_set_length

        celf_ep = [mep_k_prod, mep_i_node, mep_mg1, mep_prev_best, mep_mg2, mep_flag]
        celf_sequence.append(celf_ep)
        for celf_item in celf_sequence:
            if celf_ep[2] >= celf_item[2]:
                celf_sequence.insert(celf_sequence.index(celf_item), celf_ep)
                celf_sequence.pop()
                break

        mep_g = celf_sequence.pop(0)
        mep_k_prod, mep_i_node, mep_mg1, mep_prev_best, mep_mg2, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3], mep_g[4], mep_g[5]

    pro_acc, pro_k_list_acc, pnn_k_list_acc = 0.0, [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
    for _ in range(eva_monte_carlo):
        pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
        pro_acc += pro
        for kk in range(num_product):
            pro_k_list_acc[kk] += pro_k_list[kk]
            pnn_k_list_acc[kk] += pnn_k_list[kk]
    now_profit = round(pro_acc / eva_monte_carlo, 4)
    for kk in range(num_product):
        profit_k_list[kk] += round(pro_k_list_acc[kk] / eva_monte_carlo, 4)
        pnn_k_list_acc[kk] = round(pnn_k_list_acc[kk] / eva_monte_carlo, 2)
    now_budget = round(now_budget, 2)

    # -- result --
    now_num_k_seed = [len(kk) for kk in seed_set]
    result.append([now_profit, now_budget, now_num_k_seed, pnn_k_list_acc, seed_set])
    avg_profit += now_profit
    avg_budget += now_budget
    for kk in range(num_product):
        budget_k_list[kk] = round(budget_k_list[kk], 2)
        avg_num_k_seed[kk] += now_num_k_seed[kk]
        avg_num_k_pn[kk] += pnn_k_list_acc[kk]
    how_long = round(time.time() - start_time, 2)
    print("\nresult")
    print(result)
    print("\npro_k_list, budget_k_list")
    print(profit_k_list, budget_k_list)
    print("total time: " + str(how_long) + "sec")