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


def get_train_feature(item, public, author_paper):
    
    paperId, authorId, _, label = item

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

    authors = paperInfo["authors"]
        

    # TODO: deepcopy?
    relatedPapers = deepcopy(author_paper[authorId])
    
    relatedCoauthors = ""
    relatedOrgs = []
    relatedVenue = []
    relatedKeyword = []
    relatedTitleWords = []
    relatedAbstractWords = []

    
    #去除待分类论文
    if paperId in relatedPapers:
        relatedPapers.remove(paperId)
        
    for relatedPaper in relatedPapers:
        
        relatedPaperInfo = public[relatedPaper]
        #print(relatedPaperInfo)
        
            
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
    public_path = "data/v3/train/train_pub.json"
    author_profile_path = "data/v3/train/train_author.json"
    train_pub_path = "data/v3/processed/train_pub.pkl"
    
    feature_path = "data/v3/processed/samples_feature_3.pkl"
    
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    public = load_json(public_path)
    author_profile = load_json(author_profile_path)
    
    if not osp.exists(train_pub_path):
        # 重新生成，可能有点慢
        public = transform_pub(public)
        dump_json(train_pub_path, public)
    else:
        # 或者加载现有的
        public = load_json(train_pub_path)
    
    
    # 选取同名人数大于指定值的作者作为训练集
    negNum = 1
    paper_and_author = []
    author_paper_train = {}
    for name in tqdm(author_profile, desc='sampling'):
        if len(author_profile[name]) > negNum:
            for person in author_profile[name]:
                author_paper_train[person] = deepcopy(author_profile[name][person])
                for paper in author_profile[name][person]:
                    paper_and_author.append((paper, person, name))

    print(len(paper_and_author))
    

    pos_neg_sample = []
    for item in tqdm(paper_and_author):
        pos_neg_sample.extend(get_train_sample(item, author_profile, negNum=negNum))
        
    
    samples_feature = []
    for item in tqdm(pos_neg_sample):
        samples_feature.append(get_train_feature(item, public, author_paper_train))
        
    samples_all = pd.concat(samples_feature, axis=0)
    
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
                                        
    # abstract
    abstract_embedding = model.encode(samples_all['abstract'].values,batch_size=512,show_progress_bar=True)
    related_abstract_embeddings, abstract_indices = generate_embedding_list(samples_all['related_abstract'], model)
    cos_abstract_stat_sims_arr = calc_sims(abstract_embedding, related_abstract_embeddings, abstract_indices)

    del abstract_embedding, related_abstract_embeddings, abstract_indices
    _ = gc.collect()
                                        
    # keyword
    keywords_embedding = model.encode(samples_all['keywords'].values,batch_size=512,show_progress_bar=True)
    related_keywords_embeddings, keywords_indices = generate_embedding_list(samples_all['related_keywords'], model)
    cos_keywords_stat_sims_arr = calc_sims(keywords_embedding, related_keywords_embeddings, keywords_indices)

    del keywords_embedding, related_keywords_embeddings, keywords_indices
    _ = gc.collect()
                                        
    #合计
    all_stat_features = np.hstack((cos_title_stat_sims_arr, 
                                   cos_venue_stat_sims_arr,
                                   cos_abstract_stat_sims_arr,
                                   cos_keywords_stat_sims_arr,
                                   ))                                    
                                        

    dump_pickle(feature_path, all_stat_features)
    print(all_stat_features.shape)
    
    
if __name__=='__main__':
    main()