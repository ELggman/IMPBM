import json

from tqdm.auto import tqdm



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


def process(data_file, out_file, movie_set, data):
    global cnt
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            dialog = json.loads(line)

            context, resp = [], ''
            entity_list = []

            for turn in dialog:
                resp = turn['text']
                entity_link = [entity2id[entity] for entity in turn['entity_link'] if entity in entity2id]
                movie_link = [entity2id[movie] for movie in turn['movie_link'] if movie in entity2id]

                if turn['role'] == 'SEEKER':
                    context.append(resp)
                    entity_list.extend(entity_link + movie_link)
                else:
                    mask_resp = resp
                    for movie_name in turn['movie_name']:
                        start_ind = mask_resp.lower().find(movie_name.lower())
                        if start_ind != -1:
                            mask_resp = f'{mask_resp[:start_ind]}<movie>{mask_resp[start_ind + len(movie_name):]}'
                        # if movie_name in mask_resp:
                        #     mask_resp = mask_resp.replace(movie_name, '')
                        else:
                            cnt += 1

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

                    turn = {
                        'context': context,
                        'resp': mask_resp,
                        'rec': movie_link,
                        'entity': list(set(entity_list)),

                        'node_list': node_list,
                        'edges': edges,
                        'adj_shape': adj_shape,
                        'seeker_id':seeker_id
                    }
                    fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

                    context.append(resp)
                    entity_list.extend(entity_link + movie_link)
                    movie_set |= set(movie_link)


if __name__ == '__main__':
    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    movie_set = set()
    cnt = 0

    file_path = r"dbpedia_subkg.json"  
    data = read_json_file(file_path)

    process('test_data_dbpedia.jsonl', '../new_inspired_conv/test_data_processed.jsonl', movie_set, data)
    process('valid_data_dbpedia.jsonl', '../new_inspired_conv/valid_data_processed.jsonl', movie_set, data)
    process('train_data_dbpedia.jsonl', '../new_inspired_conv/train_data_processed.jsonl', movie_set, data)

    with open('../new_inspired_conv/item_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(movie_set), f, ensure_ascii=False)
    print(f'#movie: {len(movie_set)}')

    print(cnt)
