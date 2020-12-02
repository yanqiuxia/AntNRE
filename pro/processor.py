# _*_ coding: utf-8 _*_
# @Time : 2020/11/27 下午6:09 
# @Author : yanqiuxia
# @Version：V 0.1
# @File : processor.py
import json
import re
import traceback
from operator import itemgetter
from collections import defaultdict

def str_find(pattern, text):
    pattern = re.sub("\(", "\(", pattern)
    pattern = re.sub("\)", "\)", pattern)
    pattern = re.compile(pattern)
    find_texts = re.finditer(pattern, text)
    span = []
    for find_text in find_texts:
        start = find_text.span()[0]
        end = find_text.span()[1]
        span.append((start, end))

    return span

def split_spo_to_model_data(spo_path, out_path):
    """
    :param spo_path: spo文件路径
    :param out_path:输出文件路径
    :param is_find_all_entity:  是否将出现过但未标注的实体全部标注为实体
    :return:
    """
    op = open(out_path, 'w', encoding='utf-8')
    fp = open(spo_path, 'r', encoding='utf-8')

    line = fp.readline()  # 调用文件的 readline()方法
    i = 0
    while line:
        i += 1
        if i <= 0:
            continue

        json_data = json.loads(line)

        json_data = one_spo_to_model_data(json_data)
        json.dump(json_data, op, ensure_ascii=False)
        op.write('\n')

        line = fp.readline()

        if i%1000==0:
            print(i)


    fp.close()


def one_spo_to_model_data(json_data):
    entity = {}

    data = defaultdict()

    # 找到所有实体、实体对关系
    spo_list = json_data['spo_list']
    text = json_data['text']
    data['sentText'] = " ".join(list(text))

    for spo in spo_list:
        object_ = spo['object']["@value"]
        subject_ = spo['subject']
        object_type = spo['object_type']["@value"]
        subject_type = spo['subject_type']
        try:
            obj_sp_list = str_find(object_, text)
            sub_sp_list = str_find(subject_, text)
        except Exception as e:
            print(e)
            traceback.print_exc()
        else:
            if len(obj_sp_list)>1 or len(sub_sp_list)>1:
                global count
                count += 1
            if len(obj_sp_list)>=1:
                '只取1个'
                entity.update({obj_sp_list[0]: (object_, object_type)})

            if len(sub_sp_list)>=1:
                '只取1个'
                entity.update({sub_sp_list[0]: (subject_, subject_type)})

    entityMentions = []
    for k, v in entity.items():
        entityMentions.append({"text": v[0], "label": v[1], "offset": k})

    sorted_entityMentions = sorted(entityMentions, key=itemgetter('offset'))
    em_text_id_map = defaultdict()

    for i, mem in enumerate(sorted_entityMentions):
        mem['emId'] = i
        em_text_id_map.update({mem['text']:i})
    data['entityMentions'] = sorted_entityMentions

    relationMentions = []
    for spo in spo_list:
        object_ = spo['object']["@value"]
        subject_ = spo['subject']

        predicate = spo['predicate']
        mem = defaultdict()
        em1Id = em_text_id_map.get(subject_)
        em2Id = em_text_id_map.get(object_)
        if em1Id is not None and em2Id is not None:
            mem['em1Id'] = em1Id
            mem['em2Id'] = em2Id
            mem['em1Text'] = subject_
            mem['em2Text'] = object_
            mem['label'] = predicate
            relationMentions.append(mem)
        else:
            print("%s,%s not in data."%(object_, subject_))

    data['relationMentions'] = relationMentions
    return data


if __name__ == "__main__":
    ''
    count = 0
    split_spo_to_model_data('../joint_entrel_gcn/data/train_data.json', '../joint_entrel_gcn/data/train.pre.json')
    print("重复的实体个数：", count)

