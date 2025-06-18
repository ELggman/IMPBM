import json
from tqdm.auto import tqdm
from collections import defaultdict



def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data



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
                        # edges.append((neighbor, node))  
        all_nodes.update(next_level_nodes)
        current_level_nodes = next_level_nodes

    # node_list.sort(key=lambda x: (
    # x not in target_nodes, target_nodes.index(x) if x in target_nodes else len(target_nodes) + node_list.index(x)))
    # node_list.sort(key=lambda x: (
    # x in target_nodes, target_nodes.index(x) if x in target_nodes else len(target_nodes) + node_list.index(x)))


    target_node_set = set(target_nodes)
    remaining_nodes = list(all_nodes - target_node_set)
    node_list = target_nodes + remaining_nodes

    node_list = list(set(node_list))

    edges = [item for sublist in edges for item in sublist]

    return node_list, edges


def process(data_file, out_file, movie_set, data, user_item_interaction):
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            dialog = json.loads(line)

            context, resp = [], ''
            entity_list = []
            items_mentioned = []

            for turn in dialog:
                resp = turn['text']
                entity_link = [entity2id[entity] for entity in turn['entity_link'] if entity in entity2id]
                movie_link = [entity2id[movie] for movie in turn['movie_link'] if movie in entity2id]

                items_mentioned.extend(entity_link)
                items_mentioned.extend(movie_link)

                # if turn['role'] == 'SEEKER':
                #     context.append(resp)
                #     entity_list.extend(entity_link + movie_link)
                # else:

                edges = []
                adj_shape = 0
                e_list = [e for e in entity_list if str(e) in data]
                if e_list:
                    k = 3  # hops

                    
                    node_list, edges = extract_k_hop_nodes(data, e_list, k)
                    adj_shape = len(node_list)
                    node_list = list(set(node_list + entity_list))
                else:
                    node_list = entity_list

                if turn['role'] == turn['seeker_id']:
                    seeker_id = turn['recommender_id']
                else:
                    seeker_id = turn['seeker_id']

                if len(context) == 0:
                    context.append('')
                tu = {
                    'context': context,
                    'resp': resp,
                    'rec': list(set(movie_link + entity_link)),
                    'entity': list(set(entity_list)),

                    'node_list': node_list,
                    'edges': edges,
                    'adj_shape': adj_shape,
                    'seeker_id': seeker_id

                }
                fout.write(json.dumps(tu, ensure_ascii=False) + '\n')

                context.append(resp)
                entity_list.extend(entity_link + movie_link)
                movie_set |= set(movie_link)

            user_item_interaction[turn['seeker_id']].update(items_mentioned)
            user_item_interaction[turn['recommender_id']].update(items_mentioned)


if __name__ == '__main__':
    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    item_set = set()

    file_path = r"dbpedia_subkg.json"  
    data = read_json_file(file_path)
    user_item_interaction = defaultdict(set)

    process('test_data_dbpedia.jsonl', '../new_inspired_pre/test_data_processed.jsonl', item_set, data,
            user_item_interaction)
    process('valid_data_dbpedia.jsonl', '../new_inspired_pre/valid_data_processed.jsonl', item_set, data,
            user_item_interaction)
    process('train_data_dbpedia.jsonl', '../new_inspired_pre/train_data_processed.jsonl', item_set, data,
            user_item_interaction)

    user_item_interaction = {user: list(movies) for user, movies in user_item_interaction.items()}
    user_item_interaction = defaultdict(list, sorted(user_item_interaction.items()))

    with open('../new_inspired_pre/user_items_interaction.json', 'w', encoding='utf-8') as f:
        json.dump(user_item_interaction, f, ensure_ascii=False)

    with open('../new_inspired_pre/item_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(item_set), f, ensure_ascii=False)
    print(f'#item: {len(item_set)}')
