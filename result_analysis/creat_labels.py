
import pandas as pd

df_original = pd.read_excel('./data/patho_info_2018 - copy.xlsx', sheet_name='Pathological_Info').values.tolist()
label_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# label_name = [裂孔，劈裂，水肿，黄斑结构，视网膜前，玻璃体，内界膜，光感受器层，椭圆体带，神经上皮层脱离，神经上皮层反射异常，神经上皮层结构异常，RPE层脱离，RPE层反射异常，RPE层结构异常，脉络膜]
recodes = []
for row in df_original[:]:
    single_recode = dict()
    # # 诊断意见
    diagnostic = ''
    for i in label_list:
        if type(row[i]) == str:
            diagnostic += row[i]
    single_recode['check_num'] = row[0]
    # # 去掉括号
    diagnostic = list(diagnostic)
    l_bracket = []
    r_bracket = []
    # 去掉每条诊断意见中的括号
    for i, word in enumerate(diagnostic):
        if word == '（':
            l_bracket.append(i)
        if word == '）':
            r_bracket.append(i)
    assert len(l_bracket) == len(r_bracket)
    if len(l_bracket) != 0:
        if len(l_bracket) != len(r_bracket):
            print(row[0])
        else:
            for idx in range(len(l_bracket)-1, -1, -1):
                del diagnostic[l_bracket[idx]:r_bracket[idx]+1]
    sentences = ''.join(diagnostic)
    single_recode['diagnostic'] = sentences
    # label
    single_recode['label'] = ['0'] * 16
    for id, i in enumerate(label_list):
        if type(row[i]) == str:
               if i == 12: #  神经上皮层脱离
                   if '贴附' in row[i] and '脱离' not in row[i]:
                       single_recode['label'][id] = '0'
                   else:
                       single_recode['label'][id] = '1'
               elif i == 15:
                   if '脱离' in row[i]:
                       single_recode['label'][id] = '1'
                   else:
                       single_recode['label'][id] = '0'
               else:
                   single_recode['label'][id] = '1'

    recodes.append(single_recode)

with open('../2018/data/all.txt', 'r', encoding='utf-8') as f:
    recode_img = f.readlines()
    f.close()

recode_img_dict = {}
for recode in recode_img:
    recode_num = recode.split('\t', 1)[0]
    recode_img_dict[recode_num] = recode.split('\t', 3)[-1]

with open('./data/2018-original/corpus_2018_original.txt', 'w', encoding='utf-8') as f:
    for single_recode in recodes:
        if single_recode['check_num'] in recode_img_dict and single_recode['diagnostic'] != '':
            f.write(single_recode['check_num'] + '\t' + single_recode['diagnostic'] + '\t' + ' '.join(single_recode['label']) + '\t' + recode_img_dict[single_recode['check_num']])
    f.close()

with open('./data/2018-original/corpus_2018_original.txt', 'r', encoding='utf-8') as f:
    for line in f:
        print(line.split()[1])
        if line.split()[1] == '':
            print(line.split()[0])
