import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Author: Lennart Justen
# Data: 11/21/2020

# Calculate confidence thresholds based on Villon et al. 2020

# Villon, S., Mouillot, D., Chaumont, M. et al. A new method to control error rates in automated species identification
# with deep learning algorithms. Sci Rep 10, 10972 (2020). https://doi.org/10.1038/s41598-020-67573-7

df = pd.read_csv('results.csv')

tau = np.linspace(30, 100, num=1000) # range of possible thresholds

aa = len(df.loc[df['class_num']==0])  # Amblyomma americamum
dv = len(df.loc[df['class_num']==1])  # Dermacentor variabilis
ix = len(df.loc[df['class_num']==2])  # Ixodes scapularis

# calculate cc,mc,uc rates for each species at every threshold in tau
cc_a = []   # correct classification rate
mc_a = []   # misclassification rate
uc_a = []   # unsure classification rate
cc_d = []
mc_d = []
uc_d = []
cc_i = []
mc_i = []
uc_i = []
for t in tau:
    cc_a.append(len(df.loc[(df['class_num']==0) & (df['prob']>t) & (df['pred_num']==0)])/aa)
    mc_a.append(len(df.loc[(df['class_num']==0) & (df['prob']>t) & (df['pred_num']!=0)])/aa)
    uc_a.append(len(df.loc[(df['class_num']==0) & (df['prob']<t)])/aa)

    cc_d.append(len(df.loc[(df['class_num']==1) & (df['prob']>t) & (df['pred_num']==1)])/dv)
    mc_d.append(len(df.loc[(df['class_num']==1) & (df['prob']>t) & (df['pred_num']!=1)])/dv)
    uc_d.append(len(df.loc[(df['class_num']==1) & (df['prob']<t)])/dv)

    cc_i.append(len(df.loc[(df['class_num']==2) & (df['prob']>t) & (df['pred_num']==2)])/ix)
    mc_i.append(len(df.loc[(df['class_num']==2) & (df['prob']>t) & (df['pred_num']!=2)])/ix)
    uc_i.append(len(df.loc[(df['class_num']==2) & (df['prob']<t)])/ix)

# reality check: cc+mc+uc == 1 for every threshold (ignore where t=100)
total_aa = np.array(cc_a)+np.array(mc_a)+np.array(uc_a)
total_dv = np.array(cc_d)+np.array(mc_d)+np.array(uc_d)
total_ix = np.array(cc_i)+np.array(mc_i)+np.array(uc_i)


def goal_two(tau, cc, mc, uc, n, label, val):
    indices = np.where(np.array(mc[0:len(mc)-1]) < val) # All indices where mc < val. Remove t=100 value
    indices = indices[0]    # Convert from tuple to array

    # This means that no threshold t brought mc below val. In this case choose the minimum mc available
    if len(indices)==0:
        min_mc = min(mc[0:len(mc)-1])
        indices = [index for index, val in enumerate(mc[0:len(mc)-1]) if val == min_mc]

    cc_indices = [cc[i] for i in indices]   # All cc rates at the indices where mc < val
    key1 = cc_indices.index(max(cc_indices))  # Choose the index with the highest cc among all cc rates
    key2 = indices[key1]    # From key1 get the threshold index where the above conditions are satisfied
    thresh = tau[key2]

    print('-----------------------------------------------------')
    print('Goal 2: Constrain the misclassification error rate to an upper bound while maximizing the correct classification rate')
    print('LABEL: {}'.format(label))
    print('Initial accuracy={}'.format(cc[0]))
    print('Final accuracy (excluding unsure)={}'.format(((cc[key2]) * n / (n - (uc[key2] * n)))))
    print('Accuracy increase={}'.format(((cc[key2]) * n / (n - (uc[key2] * n)))-cc[0]))
    print('Threshold={}'.format(thresh))
    print('CC w/o threshold={}'.format(cc[0]))
    print('MC w/o threshold={}'.format(mc[0]))
    print('--------------')
    print('CC w/ threshold={}'.format(cc[key2]))
    print('MC w/ threshold={}'.format(mc[key2]))
    print('UC w/ threshold={}'.format(uc[key2]))
    print("Number of incorrect images moved to 'unsure': {}/{}".format(uc[key2] * n, n))


goal_two(tau, cc_a, mc_a, uc_a, aa, 'Amblyomma', val=0.02)
goal_two(tau, cc_d, mc_d, uc_d, dv, 'Dermacentor', val=0.02)
goal_two(tau, cc_i, mc_i, uc_i, ix, 'Ixodes', val=0.02)




