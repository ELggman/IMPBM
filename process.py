import json
import re

import html
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

movie_pattern = re.compile(r'@\d+')


# 从文件中读取 JSON 数据
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# 提取目标节点指向的节点和边，并直接返回节点列表、边和节点索引
def extract_k_hop_nodes(data, target_nodes, k):
    all_nodes = set(target_nodes)
    edges = []
    current_level_nodes = set(target_nodes)

    for _ in range(k):
        next_level_nodes = set()
        for node in current_level_nodes:
            if str(node) in data:
                for relation, neighbor in data[str(node)]:
                    if neighbor not in all_nodes:
                        next_level_nodes.add(neighbor)
                        edges.append((node, neighbor))
                        # edges.append((neighbor, node))  # 添加反向边以创建无向图
        all_nodes.update(next_level_nodes)
        current_level_nodes = next_level_nodes

    # node_list.sort(key=lambda x: (
    # x not in target_nodes, target_nodes.index(x) if x in target_nodes else len(target_nodes) + node_list.index(x)))
    # node_list.sort(key=lambda x: (
    # x in target_nodes, target_nodes.index(x) if x in target_nodes else len(target_nodes) + node_list.index(x)))

    # 保持 target_nodes 在前面的顺序
    target_node_set = set(target_nodes)
    remaining_nodes = list(all_nodes - target_node_set)
    node_list = target_nodes + remaining_nodes

    node_list = list(set(node_list))
    # 将edges 展平到1维
    edges = [item for sublist in edges for item in sublist]

    return node_list, edges


def process_utt(utt, movieid2name, replace_movieId):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            movie_name = movieid2name[movieid]
            movie_name = ' '.join(movie_name.split())
            return movie_name
        else:
            return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    return utt


def process(data_file, out_file, movie_set, data, user_item_interaction):
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            dialog = json.loads(line)
            if len(dialog['messages']) == 0:
                continue

            movieid2name = dialog['movieMentions']
            user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']

            movie_mentioned = []
            items_mentioned = []

            context, resp = [], ''
            entity_list = []
            messages = dialog['messages']
            turn_i = 0
            while turn_i < len(messages):
                worker_id = messages[turn_i]['senderWorkerId']
                utt_turn = []
                entity_turn = []
                movie_turn = []

                turn_j = turn_i
                while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                    utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True)
                    utt_turn.append(utt)

                    entity_ids = [entity2id[entity] for entity in messages[turn_j]['entity'] if entity in entity2id]
                    entity_turn.extend(entity_ids)

                    movie_ids = [entity2id[movie] for movie in messages[turn_j]['movie'] if movie in entity2id]
                    movie_turn.extend(movie_ids)

                    movie_mentioned.extend(movie_ids)
                    items_mentioned.extend(entity_ids)
                    items_mentioned.extend(movie_ids)

                    turn_j += 1

                utt = ' '.join(utt_turn)

                # if worker_id == user_id:
                #     context.append(utt)
                #     entity_list.append(entity_turn + movie_turn)
                # else:
                resp = utt

                context_entity_list = [entity for entity_l in entity_list for entity in entity_l]
                context_entity_list_extend = []
                # entity_links = [id2entity[id] for id in context_entity_list if id in id2entity]
                # for entity in entity_links:
                #     if entity in node2entity:
                #         for e in node2entity[entity]['entity']:
                #             if e in entity2id:
                #                 context_entity_list_extend.append(entity2id[e])
                context_entity_list_extend += context_entity_list
                context_entity_list_extend = list(set(context_entity_list_extend))

                '''
                
                '''

                node_list = []
                edges = []
                adj_shape = 0
                if context_entity_list_extend:
                    k = 3  # 跳数

                    # 提取 k 跳节点和边
                    node_list, edges = extract_k_hop_nodes(data, context_entity_list_extend, k)
                    adj_shape = len(node_list)

                if worker_id == user_id:
                    seeker_id = resp_id
                else:
                    seeker_id = user_id

                if len(context) == 0:
                    context.append('')
                turn = {
                    'context': context,
                    'resp': resp,
                    'rec': list(set(movie_turn + entity_turn)),
                    'entity': context_entity_list_extend,

                    'node_list': node_list,
                    'edges': edges,
                    'adj_shape': adj_shape,
                    'seeker_id': seeker_id,
                }
                fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

                context.append(resp)
                entity_list.append(movie_turn + entity_turn)
                movie_set |= set(movie_turn)

                turn_i = turn_j

            # user_item_interaction[user_id].update(movie_mentioned)
            # user_item_interaction[resp_id].update(movie_mentioned)
            user_item_interaction[user_id].update(items_mentioned)
            user_item_interaction[resp_id].update(items_mentioned)


if __name__ == '__main__':

    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    item_set = set()
    # with open('node2text_link_clean.json', 'r', encoding='utf-8') as f:
    #     node2entity = json.load(f)
    file_path = r""  # 替换为实际文件路径
    data = read_json_file(file_path)

    user_item_interaction = defaultdict(set)

    process('valid_data_dbpedia.jsonl', 'new/valid_data_processed.jsonl', item_set, data, user_item_interaction)
    process('test_data_dbpedia.jsonl', 'new/test_data_processed.jsonl', item_set, data, user_item_interaction)
    process('train_data_dbpedia.jsonl', 'new/train_data_processed.jsonl', item_set, data, user_item_interaction)

    user_item_interaction = {user: list(movies) for user, movies in user_item_interaction.items()}
    user_item_interaction = defaultdict(list, sorted(user_item_interaction.items()))

    with open('new/user_items_interaction.json', 'w', encoding='utf-8') as f:
        json.dump(user_item_interaction, f, ensure_ascii=False)

    with open('new/item_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(item_set), f, ensure_ascii=False)
    print(f'#item: {len(item_set)}')
