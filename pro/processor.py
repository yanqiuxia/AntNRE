# _*_ coding: utf-8 _*_
# @Time : 2020/11/27 下午6:09 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : processor.py
import json
import numpy as np
import re
import json
import pickle
import numpy as np
from collections import Counter
import math


def str_find(pattern,text):
    span = []
    idx_start = -2
    idx_end = 0
    while idx_start != -1:
        idx_start = text.find(pattern, idx_end)
        idx_end = idx_start + len(pattern)
        span.append((idx_start, idx_end))
    return span[:-1]

def split_spo_to_model_data(train_path, dev_path, spo_path, max_len, is_find_all_entity=False):
    """
    将输入的spolist按照二八划分为训练集和验证集，并按照模型需要的数据格式保存
    :param train_path: 训练集路径
    :param dev_path: 验证集路径
    :param spo_path: spo文件路径
    :param max_len: 可以处理的最大长度
    :param is_find_all_entity:  是否将出现过但未标注的实体全部标注为实体
    :return:
    """
    lens =[]
    with open(train_path,'w',encoding='utf-8') as f_train:
        with open(dev_path,'w',encoding='utf-8') as op:
            with open(spo_path, 'r', encoding='utf-8') as fp:
                all_data = json.load(fp)
                new_index = np.random.permutation(len(all_data))
                dev_start_index = math.floor(0.8*len(all_data))
                for index in new_index:
                    line = all_data[index]
                    text = line['text'].lower().replace('“', '"').replace('”', '"')
                    text = re.sub('\n', '', text)
                    lens.append(len(text))
                    spo_list = line['spo_list']
                    entities = line['entities']
                    label, span_label, current_span, cur_rel_labels, cur_candi_rels, cur_candi_rels_index = \
                        one_spo_to_model_data(text, spo_list, entities, max_len, is_find_all_entity)

                    if index <= dev_start_index:
                        f_train.write(json.dumps({'text':text, 'label':label, 'label_span':span_label, 'entity_span': current_span, 'rel':cur_rel_labels,'rel_span':cur_candi_rels, 'rel_span_index':cur_candi_rels_index},ensure_ascii = False)+'\n')
                    else:
                        op.write(json.dumps({'text':text, 'label':label, 'label_span':span_label,'entity_span': current_span, 'rel':cur_rel_labels,'rel_span':cur_candi_rels, 'rel_span_index':cur_candi_rels_index},ensure_ascii = False)+'\n')
    print(Counter(lens))

def one_spo_to_model_data(text, spo_list, entities, max_len, is_find_all_entity):
    entity = {}
    origin_entity = {}
    relations = {}
    # 找到所有实体、实体对关系
    for spo in spo_list:
        object_ = spo['object'].lower()
        subject_ = spo['subject'].lower()
        object_type = spo['object_type']
        subject_type = spo['subject_type']
        subject_start_index = spo['subject_index']
        subject_end_index = subject_start_index + len(subject_)
        object_start_index = spo['object_index']
        object_end_index = object_start_index + len(object_)
        predicate = spo['predict']
        if subject_end_index > max_len or object_end_index > max_len:
            continue
        direction = "-->"
        if subject_start_index > object_start_index:  # 保持实体的span是由小到大排序
            direction = "<--"
            relations[(
            (object_start_index, object_end_index), (subject_start_index, subject_end_index))] = predicate + direction
        else:
            relations[(
                (subject_start_index, subject_end_index), (object_start_index, object_end_index))] = predicate + direction
        origin_entity.update({(subject_start_index, subject_end_index): (subject_, subject_type)})
        origin_entity.update({(object_start_index, object_end_index): (object_, object_type)})
        if is_find_all_entity:
            obj_sp_list = str_find(object_, text)
            sub_sp_list = str_find(subject_, text)
            for sp in obj_sp_list:
                entity.update({sp: (object_, object_type)})
            for sp in sub_sp_list:
                entity.update({sp: (subject_, subject_type)})

    # 将原数据集中不存在关系的实体添加进来
    for en in entities:
        start_index = en['start_index']
        end_index = start_index + len(en['text'])
        if end_index > max_len:
            continue
        if (start_index, end_index) not in origin_entity:
            en_type = en['type']
            en_text = en['text']
            origin_entity.update({(start_index, end_index): (en_text, en_type)})
            if is_find_all_entity:
                en_list = str_find(en_text, text)
                for sp in en_list:
                    entity.update({sp: (en_text, en_type)})

    # 实体更新（防止实体重复）
    # 删除被包含的实体
    entity_sp_pop_list = []
    for sp in entity:
        for sp_ in entity:
            if sp[0] == sp_[0] and sp[1] == sp_[1]:  # 本身不变
                pass
            elif sp[0] >= sp_[0] and sp[1] <= sp_[1]:  # sp_包含sp
                entity_sp_pop_list.append(sp)
    entity.update(origin_entity)  # 确保原有标注的实体不被覆盖
    for i in list(set(entity_sp_pop_list)):
        del entity[i]

    # 标签maker
    label = ['O'] * len(text)
    span_label = ['O'] * len(text)
    for sp in entity:
        ent_type = entity[sp][1]
        label[sp[0]] = 'B-' + ent_type
        label[sp[0] + 1:sp[1]] = ['I-' + ent_type for i in range(sp[1] - sp[0] - 1)]
        span_label[sp[0]] = 'B'
        span_label[sp[0] + 1:sp[1]] = ['I' for i in range(sp[1] - sp[0] - 1)]

    # span maker
    current_span = []
    current_span_index = {}
    i = 0
    for sp in entity:
        current_span.append((sp[0], sp[1]))
        current_span_index[sp] = i
        i += 1
    # relation maker
    cur_candi_rels = []
    cur_candi_rels_index = []
    cur_rel_labels = []
    for e1 in current_span:
        for e2 in current_span:
            # 保证实体的index由小到大排序，且实体间没有重叠
            if e1[0] > e2[0]:
                continue
            if e1[1] > e2[0]:
                continue
            cur_candi_rels.append((e1[0], e1[1], e2[0], e2[1]))
            cur_candi_rels_index.append((current_span_index[(e1[0], e1[1])], current_span_index[(e2[0], e2[1])]))
            if (e1, e2) in relations:
                cur_rel_labels.append(relations[(e1, e2)])
            else:
                cur_rel_labels.append('None')  # 不存在关系的实体之间的关系为None
    return label, span_label, current_span, cur_rel_labels, cur_candi_rels, cur_candi_rels_index
if __name__ == "__main__":
    ''