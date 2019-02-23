for data_setting in [1, 2]:
    data_set_name = "email_undirected" * (data_setting == 1) + "WikiVote_directed" * (data_setting == 2) + \
                    "NetPHY_undirected" * (data_setting == 3)
    for pps in [1, 2, 3]:
        profit = []
        for prod_setting in [1, 2]:
            for prod_setting2 in [1, 2, 3]:
                product_name = "r1p3n" + str(prod_setting) + "a" * (prod_setting2 == 2) + "b" * (prod_setting2 == 3)
                for wpiwp in [bool(0), bool(1)]:
                    max_m = 6
                    for m in range(1, max_m + 1):
                        model_name = "mngic" * (m == 1) + "mhdic" * (m == 2) + "mric" * (m == 3) + "mhadic" * (m == 4) + "mpmisic" * (m == 5) + "mtoic" * (m == 6) + "_pps"

                        try:
                            result_name = "result/r_" + data_set_name + "/" + model_name + str(pps) + "_wpiwp" * wpiwp \
                                          + "/" + model_name + str(pps) + "_wpiwp" * wpiwp + "_" + product_name + "/1profit.txt"
                            print(result_name)

                            with open(result_name) as f:
                                for lnum, line in enumerate(f):
                                    if lnum == 0:
                                        profit.append(line)
                                    else:
                                        break
                            f.close()
                        except FileNotFoundError:
                            profit.append("")
                            continue

        fw = open("result/r_" + data_set_name + "/pps" + str(pps) + "_comparison_profit.txt", 'w')
        for lnum, line in enumerate(profit):
            if lnum % (max_m * 2) == 0 and lnum != 0:
                fw.write("\n" * 5)
            fw.write(str(line) + "\n")
        fw.close()