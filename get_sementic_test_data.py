import random
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import re
import gc
import random
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from utils import *
from transform import transform_pub

def cleanName(dirtyName):
    
    name = dirtyName.lower()
    name = name.replace('\xa0', ' ')
    name = name.replace('.', ' ')
    name = name.replace('dr.', '')
    name = name.replace('dr ', '')
    name = name.replace(' 0001', '')
    temp = name.split(' ')
    if len(temp) == 2:
        if '-' in temp[1] and '-' not in temp[0]:
            a = temp[1]
            temp[1] = temp[0]
            temp[0] = a
    k = []
    for t in temp:
        if t != '' and t != ' ':
            k.append(t)
    name = '_'.join(k)
    name = name.replace('-', '')
    return name

def get_train_sample(item, author_profile, negNum=1):
    sample = []
    paper, author, name = item
    sample.append((paper, author, name, 1))

    candidateList = list(author_profile[name].keys())
    candidateList.remove(author)
    negPeople = random.sample(candidateList, negNum)
    for negPerson in negPeople:
        sample.append((paper, negPerson, name, 0))
    return sample


def get_stat_features(arr):
    if not arr.size:
        return [0., 0., 0., 0.]
    else:
        mean_arr = np.mean(arr)
        max_arr = np.max(arr)
        min_arr = np.min(arr)
        std_arr = np.std(arr)
        stats_arr = [mean_arr, max_arr, min_arr, std_arr]
        return stats_arr


def get_test_feature(item, public, author_paper, whole_public):
    
    paperId, authorId = item

    paperInfo = public[paperId]
    #print(paperInfo)

    
    title = paperInfo["title"]
    # 标题转换为句子
    if title == None:
        title_sentence = ""
    else:
        title_sentence = " ".join(title)
        title_sentence = title_sentence.lower()
        
    #print(f'title_sentence : {title_sentence}')
    
    #title_sentence_embedding = model.encode(title_sentence)
    

    venue = paperInfo["venue"]
    
    if venue == None:
        venue_sentence = ""
    else:
        venue_sentence = venue.lower()
    
    

#     print(f'coauthors_sentence : {coauthors_sentence}')
       
    
    keyword = paperInfo.get("keywords", [])
    
    if keyword == None:
        keyword_sentence = ""
    else:
        keyword_sentence = ""
        for item in keyword:
            tmpkeyword = ' '.join(item)
            keyword_sentence += tmpkeyword
        keyword_sentence = keyword_sentence.lower()
        
#    print(f'keyword_sentence : {keyword_sentence}')
    
        
        
    abstract = paperInfo.get("abstract", [])
    if abstract == None:
        abstract_sentence = ""
    else:
        abstract_sentence = ' '.join(abstract)
        abstract_sentence = abstract_sentence.lower()
        
    #print(f'abstract_sentence : {abstract_sentence}')

        
    #print(orgs)


    # TODO: deepcopy?
    relatedPapers = deepcopy(author_paper[authorId])
    
    relatedCoauthors = ""
    relatedOrgs = []
    relatedVenue = []
    relatedKeyword = []
    relatedTitleWords = []
    relatedAbstractWords = []

    # TODO: deepcopy?
    relatedPapers = deepcopy(author_paper[authorId]["pubs"])
    # 去除待分类论文
    if paperId in relatedPapers:
        relatedPapers.remove(paperId)
    for relatedPaper in relatedPapers:
        
        relatedPaperInfo = whole_public[relatedPaper]
        
            
        #print(f'relatedOrgs : {relatedOrgs}')

        #期刊
        relatedVenue.append(relatedPaperInfo.get("venue", "none").lower())
        
        #print(f'relatedVenue : {relatedVenue}')
            
        
        #关键词
        keyword_sentence = ""

        item = relatedPaperInfo.get("keywords", [])
        
        if item == None:
            relatedKeyword.append(keyword_sentence)
        else:
            tmpKeywords = []
            for tmp in item:
                tmpkeyword = ' '.join(tmp)
                tmpKeywords.append(tmpkeyword)
        
        relatedKeyword.append(' '.join(tmpKeywords))
        

        #题目
        tempTitle = relatedPaperInfo["title"]
        if tempTitle == None:
            relatedTitleWords.append("")
        else:
            relatedTitleWords.append(' '.join(tempTitle).lower())
            
        #print(f'relatedTitleWords : {relatedTitleWords}')
        
        # 摘要
        tempAbstracts = relatedPaperInfo.get("abstract", [])
        if tempAbstracts == None:
            relatedAbstractWords.append("")
        else:
            relatedAbstractWords.append(' '.join(tempAbstracts).lower())
            
        #print(f'relatedAbstractWords : {relatedAbstractWords}')

    feature = pd.DataFrame()
    feature['title'] = [title_sentence]
    feature['venue'] = [venue_sentence]
    feature['abstract'] = [abstract_sentence]
    feature['keywords'] = [keyword_sentence]
    feature['related_title'] = [relatedTitleWords]
    feature['related_venue'] = [relatedVenue]
    feature['related_abstract'] = [relatedAbstractWords]
    feature['related_keywords'] = [relatedKeyword]
    
    return feature

def generate_embedding_list(items, model):
    indices = []
    stacked_features = []
    for arr in items:
        indices.append(len(arr))
        stacked_features.extend(arr)
    
    print(len(stacked_features))
    
    stacked_embeddings = model.encode(stacked_features,batch_size=512,show_progress_bar=True)
    
    return stacked_embeddings, indices

def calc_sims(embeddings, related_embeddings, indices):
    cos_stat_sims_list = []

    tmp_idx = 0
    for num, idx in enumerate(tqdm(indices)):

        related_embeddings_tmp = related_embeddings[tmp_idx: (tmp_idx + idx)]

        if not related_embeddings_tmp.size:
            cos_stat_sims_list.append([0., 0., 0., 0.])
            tmp_idx += idx
            continue

        else:
            cos_sims = cosine_similarity(embeddings[num].reshape(1, -1), related_embeddings_tmp)
            cos_stat_sims = get_stat_features(cos_sims)
            cos_stat_sims_list.append(cos_stat_sims)

            tmp_idx += idx
            
    cos_stat_sims_arr = np.array(cos_stat_sims_list)
    return cos_stat_sims_arr

def main():
    
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2',device='cuda:0')
    
    val_data_path = 'data/v3/cna_test_data/cna_test_unass.json'
    val_public_path = 'data/v3/cna_test_data/cna_test_unass_pub.json'
    whole_public_path = 'data/v3/cna_data/whole_author_profiles_pub.json'
    whole_author_profiles_path = 'data/v3/cna_data/whole_author_profiles.json'
    val_pub_path = "data/v3/processed/test_pub.pkl"
    
    classifySet_all_NIL_path = "data/v3/processed/test_features_nil_3.pkl"
    paper_ids_NIL_path = "data/v3/processed/test_paper_ids_nil_3.pkl"
    candidate_all_NIL_path = "data/v3/processed/test_candidate_all_nil_3.pkl"
    
    val_data = load_json(val_data_path)
    val_public = load_json(val_public_path)
    whole_public = load_json(whole_public_path)
    whole_author_profiles = load_json(whole_author_profiles_path)
    
    if not osp.exists(val_pub_path):
        # 重新生成，可能有点慢
        val_public = transform_pub(val_public)
        dump_json(val_pub_path, val_public)
    else:
        # 或者加载现有的
        val_public = load_json(val_pub_path)
        
    author_data = defaultdict(list)
    for item in whole_author_profiles:
        name = cleanName(whole_author_profiles[item]["name"])
        author_data[name].append(item)
        
    NIL = []
    classifySet_all = []
    indices_all = []
    candidate_all = []
    paper_ids = [] 

    for item in tqdm(val_data): ##########
        paperId, index = item.split('-')
        paperInfo = val_public[paperId]
        name = paperInfo["authors"]
        name = cleanName(name[int(index)]["name"])

        candidate = []
        candidate = author_data.get(name, [])
        if not candidate:
            name = name_reverse(name)
            candidate = author_data.get(name, [])

        if not candidate:
            NIL.append(item)
            continue

        ################################################    

        classifySet = []

        for personId in candidate:
            exam = (paperId, personId) # 用 item的形式取代字符串加'-'连接
            temp = get_test_feature(exam, val_public, whole_author_profiles, whole_public)
            classifySet.append(temp)
        
        indices_all.append(len(classifySet))
            
        classifySet = pd.concat(classifySet, axis=0)
        #print(classifySet.shape)

        classifySet_all.append(classifySet)
        paper_ids.append(paperId)
        candidate_all.append(candidate)
        
        #NIL.append(item) ############       
 
    samples_all = pd.concat(classifySet_all, axis=0)
    print(f'The indice is {indices_all[0]}')
    
    del classifySet_all
    _ = gc.collect()
    
    
     # title
    title_embedding = model.encode(samples_all['title'].values,batch_size=512,show_progress_bar=True)
    related_title_embeddings, title_indices = generate_embedding_list(samples_all['related_title'], model)
    cos_title_stat_sims_arr = calc_sims(title_embedding, related_title_embeddings, title_indices)
    
    del title_embedding, related_title_embeddings, title_indices
    _ = gc.collect()
    
    # venue
    venue_embedding = model.encode(samples_all['venue'].values,batch_size=512,show_progress_bar=True)
    related_venue_embeddings, venue_indices = generate_embedding_list(samples_all['related_venue'], model)
    cos_venue_stat_sims_arr = calc_sims(venue_embedding, related_venue_embeddings, venue_indices)

    del venue_embedding, related_venue_embeddings, venue_indices
    _ = gc.collect()
    
    # keyword
    keywords_embedding = model.encode(samples_all['keywords'].values,batch_size=512,show_progress_bar=True)
    related_keywords_embeddings, keywords_indices = generate_embedding_list(samples_all['related_keywords'], model)
    cos_keywords_stat_sims_arr = calc_sims(keywords_embedding, related_keywords_embeddings, keywords_indices)

    del keywords_embedding, related_keywords_embeddings, keywords_indices
    _ = gc.collect()
                                        
    # abstract
    abstract_embedding = model.encode(samples_all['abstract'].values,batch_size=512,show_progress_bar=True)
    related_abstract_embeddings, abstract_indices = generate_embedding_list(samples_all['related_abstract'], model)
    cos_abstract_stat_sims_arr = calc_sims(abstract_embedding, related_abstract_embeddings, abstract_indices)

    del abstract_embedding, related_abstract_embeddings, abstract_indices
    _ = gc.collect()
                                        

                                        
                              
                                        
    #合计
    all_stat_features = np.hstack((cos_title_stat_sims_arr, 
                                   cos_venue_stat_sims_arr,
                                   cos_abstract_stat_sims_arr,
                                   cos_keywords_stat_sims_arr,
                                   ))
    
    tmp_idx = 0
    all_stat_features_list = []
    for _, idx in enumerate(tqdm(indices_all)):

        stat_features_tmp = all_stat_features[tmp_idx: (tmp_idx + idx)]
        
        all_stat_features_list.append(stat_features_tmp)

        tmp_idx += idx
        
    print(f'len(all_stat_features_list) = {len(all_stat_features_list)}')
    print(all_stat_features_list[0].shape)
        
        

    print(f'第一次未匹配文章数 {len(NIL)}')

    NIL2 = []
    classifySet_nil_all = []
    indices_nil_all = []

    for item in tqdm(NIL):
        paperId, index = item.split('-')
        paperInfo = val_public[paperId]
        name = paperInfo["authors"]
        name = cleanName(name[int(index)]["name"])

        # 去除逗号
        name = name_remove_comma(name)
        # 去除奇怪的0
        name = name_remove_zero(name)
        
        candidate = author_data.get(name, [])
        # 姓名翻转的处理
        if not candidate:
            candidate = author_data.get(name_reverse(name), [])

        # 其它名字的处理，包括中文名，日文名以及乱码名
        if not candidate:
            candidate_name = mapping_test.get(name, None)
            if candidate_name:
                candidate = author_data.get(candidate_name, [])

        # 名称简称的处理
        if not candidate:
            candidate_name = find_all_candidate(name, author_data) + find_all_candidate(name_reverse(name), author_data)
            if candidate_name:
                for c in candidate_name:
                    candidate.extend(author_data[c])

        if not candidate:
            NIL2.append(item)
            continue        

        ################################################    

        classifySet = []
        for personId in candidate:
            exam = (paperId, personId) # 用 item的形式取代字符串加'-'连接
            temp = get_test_feature(exam, val_public, whole_author_profiles, whole_public)
            classifySet.append(temp)

        indices_nil_all.append(len(classifySet))
            
        classifySet = pd.concat(classifySet, axis=0)            
            
        classifySet_nil_all.append(classifySet) #新的nil，后面拼接
        paper_ids.append(paperId)
        candidate_all.append(candidate)
        
    samples_nil_all = pd.concat(classifySet_nil_all, axis=0)
    print(f'The indice is {indices_nil_all[0]}')
    
    # title
    title_embedding = model.encode(samples_nil_all['title'].values,batch_size=512,show_progress_bar=True)
    related_title_embeddings, title_indices = generate_embedding_list(samples_nil_all['related_title'], model)
    cos_title_stat_sims_arr = calc_sims(title_embedding, related_title_embeddings, title_indices)
    
    del title_embedding, related_title_embeddings, title_indices
    _ = gc.collect()
    
    # venue
    venue_embedding = model.encode(samples_nil_all['venue'].values,batch_size=512,show_progress_bar=True)
    related_venue_embeddings, venue_indices = generate_embedding_list(samples_nil_all['related_venue'], model)
    cos_venue_stat_sims_arr = calc_sims(venue_embedding, related_venue_embeddings, venue_indices)

    del venue_embedding, related_venue_embeddings, venue_indices
    _ = gc.collect()
                                        
    # abstract
    abstract_embedding = model.encode(samples_nil_all['abstract'].values,batch_size=512,show_progress_bar=True)
    related_abstract_embeddings, abstract_indices = generate_embedding_list(samples_nil_all['related_abstract'], model)
    cos_abstract_stat_sims_arr = calc_sims(abstract_embedding, related_abstract_embeddings, abstract_indices)

    del abstract_embedding, related_abstract_embeddings, abstract_indices
    _ = gc.collect()
                                        
    # keyword
    keywords_embedding = model.encode(samples_nil_all['keywords'].values,batch_size=512,show_progress_bar=True)
    related_keywords_embeddings, keywords_indices = generate_embedding_list(samples_nil_all['related_keywords'], model)
    cos_keywords_stat_sims_arr = calc_sims(keywords_embedding, related_keywords_embeddings, keywords_indices)

    del keywords_embedding, related_keywords_embeddings, keywords_indices
    _ = gc.collect()
                                        
                                                                    
    #合计
    all_stat_nil_features = np.hstack((cos_title_stat_sims_arr, 
                                   cos_venue_stat_sims_arr,
                                   cos_abstract_stat_sims_arr,
                                   cos_keywords_stat_sims_arr,
                                   ))
    
    tmp_idx = 0
    all_stat_features_nil_list = []
    for _, idx in enumerate(tqdm(indices_nil_all)):

        stat_features_tmp = all_stat_nil_features[tmp_idx: (tmp_idx + idx)]
        
        all_stat_features_nil_list.append(stat_features_tmp)

        tmp_idx += idx
        
    print(f'len(all_stat_features_nil_list) = {len(all_stat_features_nil_list)}')
    print(all_stat_features_nil_list[0].shape)
    

    print(f'第二次未匹配文章数 {len(NIL2)}')  
    
    all_stat_features_list.extend(all_stat_features_nil_list) 

    dump_pickle(classifySet_all_NIL_path, all_stat_features_list)
    dump_pickle(paper_ids_NIL_path, paper_ids)
    dump_pickle(candidate_all_NIL_path, candidate_all)
    dump_pickle("data/v3/processed/NIL2.pkl", NIL2)
    
if __name__=='__main__':
    main()
