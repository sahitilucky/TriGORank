import numpy as np

Baseline_fold1 = [0.8899541168,0.7913077956,0.67741462,0.6304926,0.55]
Baseline_GO_graph_top_10_fold1 = [0.9337457765 ,0.7825473269,0.6953848485,0.6797206830,0.6]
Baseline_intersct_ws_fold1=[0.8263333286,0.7843333009,0.7407971594,0.6645086585904374,0.6]
Baseline_intersct_ws_GOgraphtop10_fold1 =[0.7917063341,0.8332357780,0.7327400188,0.64086535,0.575]

Baseline_fold2 = [ 0.3570575716,0.2973612410,0.2973911926,0.3334070278,0.35]
Baseline_GO_graph_top_10_fold2 = [0.3882047626,0.3542323523,0.3182627709,0.31526882056799493,0.325]
Baseline_intersct_ws_fold2= [ 0.5965377185,0.4533609461,0.4180539158,0.3971334713,0.325]
Baseline_intersct_ws_GOgraphtop10_fold2 =[ 0.5677211578,0.4778760632,0.4129165705,0.3941340876095334,0.325]

Baseline_fold3 = [ 0.3347092591 ,0.3222646787,0.3645591932,0.4058187451545348,0.45]
Baseline_GO_graph_top_10_fold3 = [ 0.5352543167,0.5184533931,0.4893691522,0.4735759914954853,0.425]
Baseline_intersct_ws_fold3=[ 0.5103515512,0.5066574790545515,0.505049801320715,0.46992440692803866,0.425]
Baseline_intersct_ws_GOgraphtop10_fold3 =[ 0.4269620754,0.3807024339,0.4309906202,0.42504110091050845,0.45]

Baseline_fold4 = [0.6429424283,0.5173720029,0.4655361855,0.43654076891214316,0.375]
Baseline_GO_graph_top_10_fold4 = [ 0.4864918354,0.4188878395,0.3678233536,0.4088960883739708,0.4]
Baseline_intersct_ws_fold4=[ 0.6969179698,0.5549375352,0.4971309933,0.46368394418665454,0.4]
Baseline_intersct_ws_GOgraphtop10_fold4 =[0.6618313225,0.6350566929350071,0.535040827467612,0.4938905682069485,0.425]

Baseline_fold5 = [ 0.4690000933,0.5163268862729016,0.4926912676,0.4767746844028279,0.425]
Baseline_GO_graph_top_10_fold5 = [0.5606114592,0.4734783210,0.3863155572,0.40633894080621685,0.35]
Baseline_intersct_ws_fold5=[ 0.5836175861,0.5871153665290745,0.520991665651226,0.48353187966020866,0.425]
Baseline_intersct_ws_GOgraphtop10_fold5 =[ 0.6078421520,0.6061416263097666,0.5337274906426107,0.47562503245075394,0.425]

model1 = [Baseline_fold1, Baseline_fold2, Baseline_fold3, Baseline_fold4, Baseline_fold5]
model2 = [Baseline_GO_graph_top_10_fold1,Baseline_GO_graph_top_10_fold2,Baseline_GO_graph_top_10_fold3,Baseline_GO_graph_top_10_fold4,Baseline_GO_graph_top_10_fold5]
model3 = [Baseline_intersct_ws_fold1,Baseline_intersct_ws_fold2,Baseline_intersct_ws_fold3,Baseline_intersct_ws_fold4,Baseline_intersct_ws_fold5]
model4 = [Baseline_intersct_ws_GOgraphtop10_fold1,Baseline_intersct_ws_GOgraphtop10_fold2,Baseline_intersct_ws_GOgraphtop10_fold3,Baseline_intersct_ws_GOgraphtop10_fold4,Baseline_intersct_ws_GOgraphtop10_fold5]

Lambda_intersct_ws_fold1 = [0.6618313225363274,0.5374896345599803,0.5056587145363336,0.4696362854059875, 0.4]
Lambda_intersct_ws_GOgraphtop10_fold1 = [0.6331992944486872,0.5819888652118165,0.608986723560918,0.5546209201380317,0.525]
Lambda_intersct_ws_fold2 = [0.5174612499126158,0.4788774511243548,0.4148307031007036,0.3601045655634836,0.275]
Lambda_intersct_ws_GOgraphtop10_fold2 = [0.1635413866202963,0.10554427104660292,0.08110608757909943,0.06699646598001112,0.05]
Lambda_intersct_ws_fold3 = [0.568003577662161,0.47661129013192677,0.4359876664690158,0.41248420372352707,0.35]
Lambda_intersct_ws_GOgraphtop10_fold3 = [0.6431211415940209,0.5586418841998644,0.4753424232496303,0.4612706748044995,0.4]
Lambda_intersct_ws_fold4 = [0.750637296052051,0.5595678783724828,0.49942924586202336,0.46519918122884946,0.375]
Lambda_intersct_ws_GOgraphtop10_fold4 = [0.4697557950008693,0.4859020255291319,0.48853578708376294,0.4204894314166611,0.375]
Lambda_intersct_ws_fold5 = [0.5224955967919311,0.4850010824780547,0.42033583254635887,0.3831185766330375,0.3]
Lambda_intersct_ws_GOgraphtop10_fold5 = [0.4315420454124844,0.3862291562078011,0.41446611508223574,0.37706652693906306,0.35]

model5 = [Lambda_intersct_ws_fold1, Lambda_intersct_ws_fold2, Lambda_intersct_ws_fold3, Lambda_intersct_ws_fold4, Lambda_intersct_ws_fold5]
model6 = [Lambda_intersct_ws_GOgraphtop10_fold1, Lambda_intersct_ws_GOgraphtop10_fold2, Lambda_intersct_ws_GOgraphtop10_fold3, Lambda_intersct_ws_GOgraphtop10_fold4, Lambda_intersct_ws_GOgraphtop10_fold5]


NDCGS = ["NDCG@10", "NDCG@20", "NDCG@30" ,"NDCG@40", "Prec@40"]
for i in range(5):
    performaces1,performaces2,performaces3,performaces4, performaces5, performaces6 = [],[],[],[],[],[]
    improvement21s, improvement31s,improvement41s = [], [],[]
    for j in range(5):
        performaces1 += [model1[j][i]]
        performaces2 += [model2[j][i]]
        performaces3 += [model3[j][i]]
        performaces4 += [model4[j][i]]
        performaces5 += [model5[j][i]]
        performaces6 += [model6[j][i]]
        improvement21 = (model2[j][i]-model1[j][i])*100/(model1[j][i])
        improvement31 = (model3[j][i]-model1[j][i])*100/(model1[j][i])
        improvement41 = (model4[j][i]-model1[j][i])*100/(model1[j][i])
        improvement21s += [improvement21]
        improvement31s += [improvement31]
        improvement41s += [improvement41]

    print ("ndcg@: baseline: ",NDCGS[i], np.average(np.array(performaces1)), np.std(np.array(performaces1)))
    print ("ndcg@: baseline+top10: ",NDCGS[i], np.average(np.array(performaces2)), np.std(np.array(performaces2)))
    print ("improvement%: baseline+top10: ",NDCGS[i], np.average(np.array(improvement21s)))
    print ("ndcg@: baseline+intsct: ",NDCGS[i], np.average(np.array(performaces3)), np.std(np.array(performaces3)))
    print ("improvement@: baseline+intsct: ",NDCGS[i], np.average(np.array(improvement31s)))
    print("ndcg@: baseline+top10+intersct: ",NDCGS[i], np.average(np.array(performaces4)), np.std(np.array(performaces4)))
    print ("improvement@: baseline+top10+intsct: ",NDCGS[i], np.average(np.array(improvement41s)))

    print("ndcg@: baseline+intersct: ", NDCGS[i], np.average(np.array(performaces5)), np.std(np.array(performaces5)))
    print("ndcg@: baseline+top10+intersct: ", NDCGS[i], np.average(np.array(performaces6)), np.std(np.array(performaces6)))

