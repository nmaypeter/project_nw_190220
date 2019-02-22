from Diffusion_NormalIC import *


class SeedSelectionTO:
    def __init__(self, g_dict, s_c_dict, prod_list, total_bud):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### total_bud: (int) the budget to select seed
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.total_budget = total_bud
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)

    def getSeedSetProfit(self, k_prod, i_node, s_set):
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

    def generateTopOrderList(self, data_name, w_list):
        ### t_o_list: (list) [#product, #node, out-degree, out-degree order, affordable out-degree, affordable out-degree order,
        #                   expected profit, expected profit order, order value, total order]
        t_o_list = [[] for _ in range(self.num_node * self.num_product)]
        iniG_g = IniGraph(data_name)

        for k in range(self.num_product):
            for i in range(self.num_node):
                t_o_list[k * self.num_node + i].append(k)
                t_o_list[k * self.num_node + i].append(i)
                t_o_list[k * self.num_node + i].append(iniG_g.getNodeOutDegree(str(i)))
                t_o_list[k * self.num_node + i].append(0)

                affordable_deg = 0
                if str(i) in self.graph_dict:
                    for ad in self.graph_dict[str(i)]:
                        if copy.deepcopy(w_list)[int(ad)] >= self.product_list[k][2]:
                            affordable_deg += 1
                t_o_list[k * self.num_node + i].append(affordable_deg)
                t_o_list[k * self.num_node + i].append(0)
                t_o_list[k * self.num_node + i].append(0.0)
                t_o_list[k * self.num_node + i].append(0)
                t_o_list[k * self.num_node + i].append(0)
                t_o_list[k * self.num_node + i].append(0)

        return t_o_list

    def insertExpectProfitOrder(self, t_o_list, s_set):
        # -- calculate expected profit for all combinations of nodes and products --
        ssto_ss = SeedSelectionTO(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget)

        for k in range(self.num_product):
            for i in range(self.num_node):
                ep = ssto_ss.getSeedSetProfit(k, str(i), s_set)
                t_o_list[k * self.num_node + i][6] = ep
                t_o_list[k * self.num_node + i][7] = 0

        return t_o_list

    @staticmethod
    def insertOrder(t_list):
        list_deg, list_aff_deg, list_ep = [], [], []
        # -- display the kinds for degree, aff_deg, ep --
        for item in t_list:
            if item[2] in list_deg:
                continue
            else:
                list_deg.append(item[2])
        for item in t_list:
            if item[4] in list_aff_deg:
                continue
            else:
                list_aff_deg.append(item[4])
        for item in t_list:
            if item[6] in list_ep:
                continue
            else:
                list_ep.append(item[6])

        # -- sort the kinds --
        list_deg = sorted(list_deg, reverse=True)
        list_aff_deg = sorted(list_aff_deg, reverse=True)
        list_ep = sorted(list_ep, reverse=True)

        # -- insert the order for the kinds --
        for item in t_list:
            for ll in list_deg:
                if item[2] == ll:
                    item[3] = list_deg.index(ll) + 1
            for ll in list_aff_deg:
                if item[4] == ll:
                    item[5] = list_aff_deg.index(ll) + 1
            for ll in list_ep:
                if item[6] == ll:
                    item[7] = list_ep.index(ll) + 1

        # -- display the kinds for total order --
        list_total_order = []
        for item in t_list:
            item[8] = item[5] + item[7]
            if item[8] in list_total_order:
                continue
            else:
                list_total_order.append(item[8])

        # -- sort the total order --
        list_total_order = sorted(list_total_order, reverse=False)

        # -- insert the order for total order --
        for item in t_list:
            for ll in list_total_order:
                if item[8] == ll:
                    item[9] = list_total_order.index(ll) + 1

        return t_list

    @staticmethod
    def constructTopOrderDict(t_o_list):
        t_dict = {}
        for item in t_o_list:
            k_prod = item[0]
            i_node = str(item[1])
            total_order = str(item[9])

            if total_order in t_dict:
                t_dict[total_order].add((k_prod, i_node))
            else:
                t_dict[total_order] = {(k_prod, i_node)}

        return t_dict

    def getTopOrderNode(self, t_o_dict, cur_bud):
        # -- get the node with top order --
        mep = [0, '-1']
        great_order = -1
        while mep[1] == '-1':
            while great_order == -1:
                for deg in list(t_o_dict.keys()):
                    if (int(deg) < great_order and great_order >= 0) or great_order < 0:
                        great_order = int(deg)

                if great_order == -1:
                    return mep, t_o_dict

                if t_o_dict[str(great_order)] == set():
                    del t_o_dict[str(great_order)]
                    great_order = -1

            if t_o_dict[str(great_order)] == set():
                del t_o_dict[str(great_order)]
                great_order = -1
                continue

            mep[0], mep[1] = choice(list(t_o_dict[str(great_order)]))
            t_o_dict[str(great_order)].remove((mep[0], mep[1]))

            if self.seed_cost_dict[mep[1]] + cur_bud > self.total_budget:
                mep[1] = '-1'

        return mep, t_o_dict


if __name__ == "__main__":
    data_set_name = "email_undirected"
    product_name = "r1p3n1"
    total_budget = 10
    pp_strategy = 1
    whether_passing_information_without_purchasing = bool(0)

    iniG = IniGraph(data_set_name)
    iniW = IniWallet(data_set_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()[1]
    graph_dict = iniG.constructGraphDict()
    product_list = iniP.getProductList()[0]
    wallet_list = iniW.getWalletList(product_name)
    num_node = len(seed_cost_dict)
    num_product = len(product_list)

    # -- initialization for each budget --
    start_time = time.time()

    ssto = SeedSelectionTO(graph_dict, seed_cost_dict, product_list, total_budget)
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

    top_order_list = ssto.generateTopOrderList(data_set_name, copy.deepcopy(wallet_list))
    top_order_list = ssto.insertExpectProfitOrder(top_order_list, [set() for _ in range(num_product)])
    top_order_list = ssto.insertOrder(top_order_list)
    top_order_dict = ssto.constructTopOrderDict(top_order_list)

    mep_g, top_order_dict = ssto.getTopOrderNode(top_order_dict, now_budget)
    mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    # -- main --
    while now_budget < total_budget and mep_i_node != '-1':
        seed_set[mep_k_prod].add(mep_i_node)

        budget_k_list[mep_k_prod] += seed_cost_dict[mep_i_node]
        now_budget += seed_cost_dict[mep_i_node]

        mep_g, top_order_dict = ssto.getTopOrderNode(top_order_dict, now_budget)
        mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    # fw = open("top_order_list.txt", 'w')
    # for item in top_order_list:
    #     item_kk_ii = ""
    #     t = 0
    #     for kk_ii in item:
    #         t += 1
    #         item_kk_ii += str(kk_ii) + "\t"
    #         if t % 2 == 0:
    #             item_kk_ii += "\t"
    #     fw.write(item_kk_ii + "\n")

    pro_acc, pro_k_list_acc, pnn_k_list_acc = 0.0, [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
    for _ in range(100):
        pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
        pro_acc += pro
        for kk in range(num_product):
            pro_k_list_acc[kk] += pro_k_list[kk]
            pnn_k_list_acc[kk] += pnn_k_list[kk]
    pro_acc = round(pro_acc / 100, 2)
    for kk in range(num_product):
        profit_k_list[kk] += round(pro_k_list_acc[kk] / 100, 2)
        pnn_k_list_acc[kk] = round(pnn_k_list_acc[kk] / 100, 2)
    now_budget = round(now_budget, 2)

    # -- result --
    now_num_k_seed = [len(kk) for kk in seed_set]
    result.append([pro_acc, now_budget, now_num_k_seed, pnn_k_list_acc, seed_set])
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