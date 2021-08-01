import random
import numpy as np
from copy import deepcopy
from collections import defaultdict
from pyjarowinkler import distance

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
        
def entropy(dic):
    count = []
    total = 0
    ans = 0
    for item in dic:
        count.append(dic[item])
        total += dic[item]
    prob = [item/total for item in count]
    for item in prob:
        ans += -1 * item * round(np.log(item), 6)
    return ans

def cleanYear(dic):
    count = []
    largest = 0
    smallest = 3000
    err = 0
    ans = defaultdict(int)
    for item in dic:
        if item == '':
            continue
        
        try:
            item = int(item)
        except:
            print(item)
        count.append(dic[item])
        ans[item] = dic[item]
        if item > largest:
            largest = item
        if item < smallest:
            smallest = item
    assert len(count), dic
    mean = np.mean(count)
    if smallest < largest:
        add = np.random.randint(smallest, largest, err)
    else:
        add = [smallest] * err
    for data in add:
        ans[data] += 1
    count = []
    for item in ans:
        count.append(ans[item])
    assert len(count), dic
    mean = np.mean(count)
    for item in ans:
        ans[item] = 1/(1 + np.abs(mean - ans[item]))
    
    return ans

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

    title = paperInfo["title"]
    authors = paperInfo["authors"]
    venue = paperInfo["venue"]
    year = paperInfo.get("year", 0)
    keyword = paperInfo.get("keywords", [])
    if keyword == None:
        keyword = []
    abstract = paperInfo.get("abstract", [])
    if abstract == None:
        abstract = []
    # 这里不对待查询作者进行删除
    # 对待查询论文进行预处理，获取其相关特征
    # 特征包括titleLength、words、coauthors、coauthorsLength、coauthorsDiffLength、
    #        orgs、orgsDiffLength、venu、year、keywords、abstractLength
    # 题目长度、题目词频
    titleLength = len(title)
    title_words = defaultdict(int)
    for item in title:
        title_words[item.lower()] += 1

    # 这篇paper的合作者个数、频率
    coauthors = defaultdict(int)
    coauthorsLength = 0
    for item in authors:
        if item["name"] != '':
            coauthors[cleanName(item["name"])] += 1
            coauthorsLength += 1
    coauthorsDiffLength = len(coauthors) #这篇paper的合作者的数量(unique)

    # 合作机构频率
    orgs = defaultdict(int)
    for item in authors:
        orgs[item.get("org", "").lower()] += 1
    orgsDiffLength = len(orgs)

    # 发表年份可能不存在

    #关键词词频
    keywords = defaultdict(int)
    for item in keyword:
        item = ' '.join(item)
        if item != '':
            keywords[item.lower()] += 1
    
    #摘要长度、词频
    abstractLength = len(abstract)
    abstract_words = defaultdict(int)
    for item in abstract:
        abstract_words[item.lower()] += 1

    relatedTitleLength = 0  # 题目平均长度 1
    relatedCoauthorLength = 0  # 合作者数量1
    relatedCoauthorDiffLength = 0  # 合作者人数1
    relatedOrgsDiffLength = 0  # 合作机构个数1
    relatedAbstractLength = 0  # 摘要长度
    relatedCoauthors = defaultdict(int)  # 合作者1
    relatedOrgs = defaultdict(int)  # 合作机构1
    relatedVenue = defaultdict(int)  # 发表期刊1
    relatedYear = defaultdict(int)  # 发表年份1
    relatedKeyword = defaultdict(int)  # 关键词1
    relatedTitleWords = defaultdict(int)  # 题目出现词
    relatedAbstractWords = defaultdict(int)  # 摘要出现词
    relatedOrgEntropy = 0

    # TODO: 改用deppcopy?
    relatedPapers = deepcopy(author_paper[authorId])
    
    #去除待分类论文
    if paperId in relatedPapers:
        relatedPapers.remove(paperId)
        
    for relatedPaper in relatedPapers:
        relatedPaperInfo = public[relatedPaper]
        
        tempAuthors = relatedPaperInfo["authors"]
        tempOrgs = defaultdict(int)
        for tempName in tempAuthors:
            #机构、合作者频率
            #合作者数量
            relatedOrgs[tempName.get("org", "")] += 1
            relatedCoauthors[cleanName(tempName["name"])] += 1
            tempOrgs[tempName.get("org", "")] += 1
            relatedCoauthorLength += 1
        #机构平均散度
        relatedOrgEntropy += entropy(tempOrgs)
        #年份
        tmp_year = relatedPaperInfo.get("year", 0)
        if tmp_year:
            relatedYear[tmp_year] += 1
        #期刊
        relatedVenue[relatedPaperInfo.get("venue", "none")] += 1
        #关键词
        for item in relatedPaperInfo.get("keywords", []):
            item = ' '.join(item)
            if item != '':
                relatedKeyword[item.lower()] += 1
        #题目长度
        tempTitle = relatedPaperInfo["title"]
        relatedTitleLength += len(tempTitle)
        for item in tempTitle:
            relatedTitleWords[item.lower()] += 1
        # 摘要
        tempAbstracts = relatedPaperInfo.get("abstract", [])
        if tempAbstracts == None:
            tempAbstracts = []
        
        relatedAbstractLength = len(tempAbstracts)
        for item in tempAbstracts:
            relatedAbstractWords[item.lower()] += 1

    relatedTitleLengthAve = relatedTitleLength / (len(relatedPapers) + 1)
    relatedCoauthorLengthAve = relatedCoauthorLength / (len(relatedPapers) + 1)
    relatedAbstractLengthAve = relatedAbstractLength / (len(relatedPapers) + 1)
    relatedCoauthorDiffLength = len(relatedCoauthors)
    relatedOrgsDiffLength = len(relatedOrgs)
    relatedOrgsDiffLengthAve = len(relatedOrgs) / (len(relatedPapers) + 1)
    relatedOrgEntropyAve = relatedOrgEntropy / (len(relatedPapers) + 1)
    
    # 根据待查询论文与待查询作者的相关信息生成特征
    
    ################ 1、标题相关 ################
    
    # -1、该论文标题长度
    scoreTitle1 = titleLength
    # -2、待查询论文标题平均长度
    scoreTitle2 = relatedTitleLengthAve
    # -3、该论文标题长度/待查询论文标题平均长度
    scoreTitle3 = titleLength / (relatedTitleLengthAve + 1)
    
    # -4、该论文和待查询论文标题重合的词数
    t_Words = set(title_words.keys())
    rt_Words = set(relatedTitleWords.keys())
    scoreTitle4 = len(t_Words & rt_Words)
    
    # -5、该论文标题词在待查询论文标题出现次数
    scoreTitle5 = 0
    for item in title_words:
        scoreTitle5 += relatedTitleWords.get(item, 0)
    
    # -6、待查询论文标题词平均出现次数
    scoreTitle6 = 0
    for item in relatedTitleWords:
        scoreTitle6 += relatedTitleWords[item]
    scoreTitle6 /= (len(relatedPapers) + 1)
    
    # -7、5和6比值
    scoreTitle7 = scoreTitle5/(scoreTitle6 + 1)
    # -8、4和1比值
    scoreTitle8 = scoreTitle4/(scoreTitle1 + 1)
    # -9、4和2比值
    scoreTitle9 = scoreTitle4/(scoreTitle2 + 1)
    
    
    #标题特征汇总 9
    scoreTitles = [scoreTitle1, scoreTitle2, scoreTitle3, scoreTitle4, 
                   scoreTitle5, scoreTitle6, scoreTitle7,scoreTitle8, scoreTitle9]
    
    
    ################ 2、合作者相关 ################
    
    relatedCoauthorsSet = set(relatedCoauthors.keys())
    coauthorsSet = set(coauthors.keys())
    
    # -1、待查询论文中合作者数
    scoreAuthor1 = relatedCoauthorDiffLength
    # -2、待查询论文总合作次数
    scoreAuthor2 = relatedCoauthorLength  
    # -3、该论文合作者数
    scoreAuthor3 = coauthorsDiffLength
    # -4、该论文和待查询论文重合的合作者数
    scoreAuthor4 = len(relatedCoauthorsSet & coauthorsSet)
    # -5、待查询论文平均合作者数
    scoreAuthor5 = relatedCoauthorLengthAve
    # -6、重合的合作者的总合作次数
    scoreAuthor6 = 0
    for item in coauthorsSet:
        scoreAuthor6 += relatedCoauthors[item]
        
    scoreAuthors = [scoreAuthor1, scoreAuthor2, scoreAuthor3, scoreAuthor4, scoreAuthor5, scoreAuthor6]
    
    #作者数特征两两求比值(6)
    nums_scoreAuthors = [scoreAuthor1,scoreAuthor3,scoreAuthor4,scoreAuthor5]
    scoreAuthor_ratios = [nums_scoreAuthors[i]/(nums_scoreAuthors[j]+1) for i in range(len(nums_scoreAuthors)) for j in range(i+1,len(nums_scoreAuthors))]
    
    #作者数特征两两求比值
    scoreAuthor7 = scoreAuthor2/(scoreAuthor6 + 1)
    
    # 合作者特征汇总 6 + 1 + 6 = 13 
    scoreAuthors.append(scoreAuthor7)
    scoreAuthors.extend(scoreAuthor_ratios)


    ################ 3、期刊相关 ################

    venue_is_match = 0
    venue_wordlist = venue.split()
    venue_wordset = set(venue_wordlist)
    
    # -1、重合期刊的发表次数
    scoreVenue1 = relatedVenue.get(venue, 0)
    # -2 重合期刊的发表次数/总发表期刊次数
    temp = 0
    for item in relatedVenue:
        temp += relatedVenue[item]
    scoreVenue2 = scoreVenue1 / (temp + 1)
    
    
    #作者之前发过paper所属的期刊 期刊相似度（jaro_distance和Jaccard）
    venue_str_sim_list, venue_word_sim_list = [], []
    for rv in relatedVenue.keys():
        if rv == venue:
            venue_is_match = 1
        if not venue or not rv:
            sim_1 = 0
        else:

            sim_1 = distance.get_jaro_distance(venue,rv) #字符串比对
        
        rv_wordlist = rv.split()
        rv_wordset = set(rv_wordlist)
        
        sim_2 = len(venue_wordset & rv_wordset)/(len(venue_wordset | rv_wordset) + 1) #词比对
        
        venue_str_sim_list.append(sim_1)
        venue_word_sim_list.append(sim_2)
        
    venue_str_sim_arr = np.array(venue_str_sim_list)
    venue_word_sim_arr = np.array(venue_word_sim_list)
    
    # 作者是否发过这个期刊
    scoreVenue3 = venue_is_match
    
    # 一些统计特征 (min/max/mean/std)
    venue_str_sim_stats_arr = get_stat_features(venue_str_sim_arr)
    venue_word_sim_stats_arr = get_stat_features(venue_word_sim_arr)
    
    # 合作者特征汇总 3 + 4 + 4 = 11
    scoreVenues = [scoreVenue1, scoreVenue2, scoreVenue3]
    scoreVenues.extend(venue_str_sim_stats_arr)
    scoreVenues.extend(venue_word_sim_stats_arr)
    

    ################ 4、合作机构相关 ################

    org_is_match = 0
    relatedOrgsSet = set(relatedOrgs.keys())
    orgsSet = set(orgs.keys())
    
    # -1、待查询论文合作机构数
    scoreOrg1 = orgsDiffLength
    # -2、待查询论文中作者平均合作机构
    scoreOrg2 = relatedOrgsDiffLengthAve
    # -3、待查询论文合作机构总合作次数
    scoreOrg3 = 0
    for item in orgs:
        scoreOrg3 += relatedOrgs[item]
    # -4、待查询论文合作机构散度
    scoreOrg4 = entropy(orgs)
    # -5、待查询论文中作者所有合作机构总合作次数
    scoreOrg5 = 0
    for item in relatedOrgs:
        scoreOrg5 += relatedOrgs[item]
        
    # -6、重合机构数
    scoreOrg6 = len(relatedOrgsSet & orgsSet)
    
    # -7、3和5比值
    scoreOrg7 = scoreOrg3/(scoreOrg5 + 1)
    # -8、6和1比值
    scoreOrg8 = scoreOrg6/(scoreOrg1 + 1)
    # -9、6和2比值
    scoreOrg9 = scoreOrg6/(scoreOrg2 + 1)
    
    scoreOrgs = [scoreOrg1, scoreOrg2, scoreOrg3, scoreOrg4, scoreOrg5,
                 scoreOrg6, scoreOrg7, scoreOrg8, scoreOrg9]
    
    #作者之前发过paper所属的机构 机构相似度(jaro_distance和Jaccard)
    # 这个特征好像没做好
    
    org_str_sim_list, org_word_sim_list = [], []

    for org_1 in orgs:
        for org_2 in relatedOrgs:
            if not org_1 or not org_2:
                org_sim = 0
                sim_2 = 0
            else:
  
                org_sim = distance.get_jaro_distance(org_1,org_2)
                
                org_1_set = set(org_1.split())
                org_2_set = set(org_2.split())
                sim_2 = len(org_1_set & org_2_set)/(len(org_1_set | org_2_set) + 1) #词比对
                
                if sim_2 > 0.6 or org_sim > 0.65:
                    org_is_match = 1
                
            org_str_sim_list.append(org_sim)
            org_word_sim_list.append(sim_2)

        
    org_str_sim_arr = np.array(org_str_sim_list)
    org_word_sim_arr = np.array(org_word_sim_list)
    
    # 作者之前以该机构发过paper
    scoreOrg10 = org_is_match
    
    # 一些统计特征
    org_str_sim_stats_arr = get_stat_features(org_str_sim_arr)
    org_word_sim_stats_arr = get_stat_features(org_word_sim_arr)
    
    # 机构特征汇总 9 + 1 + 4 + 4 = 18 
 
    scoreOrgs.append(scoreOrg6)
    scoreOrgs.extend(org_str_sim_stats_arr)
    scoreOrgs.extend(org_word_sim_stats_arr)


    ################ 5、关键词相关 ################

    originKeywords = set(keywords.keys())
    relatedKeywords = set(relatedKeyword.keys())
    
    # -1、该论文关键词总数
    scoreKeyword1 = len(originKeywords)
    # -2、重合关键词总数
    scoreKeyword2 = len(originKeywords & relatedKeywords)
    # -3、关键词在待查询论文总出现次数
    scoreKeyword3 = 0
    for item in relatedKeyword:
        scoreKeyword3 += relatedKeyword[item]
    # -4、1和2比值    
    scoreKeyword4 = scoreKeyword2 / (scoreKeyword1 + 1)
    
    #关键词相似度（jaro_distance和Jaccard）
    
    keyword_sim_list = []
    for kw_1 in originKeywords:
        for kw_2 in relatedKeywords:
            if not kw_1 or not kw_2:
                kw_sim = 0
            else:
                kw_sim = distance.get_jaro_distance(kw_1,kw_2)
            keyword_sim_list.append(kw_sim)

    scoreKeyword5 = scoreKeyword2 / (len(originKeywords | relatedKeywords) + 1)
        
    keyword_sim_arr = np.array(keyword_sim_list)

    keyword_sim_stats_arr = get_stat_features(keyword_sim_arr)
    
    # 关键词特征汇总 5 + 4 = 9
    scoreKeywords = [scoreKeyword1, scoreKeyword2, scoreKeyword3, scoreKeyword4, scoreKeyword5]
    scoreKeywords.extend(keyword_sim_stats_arr)
    
    ################ 6、摘要相关 ################      
    
    originWord = set(abstract_words.keys())
    relatedWord = set(relatedAbstractWords.keys())
    
    # -1、该论文摘要长度
    scoreAbstract1 = abstractLength
    # -2、待查询论文平均摘要长度
    scoreAbstract2 = relatedAbstractLengthAve
    # -3、待查询论文摘要词重合个数
    scoreAbstract3 = len(originWord & relatedWord)
    # -4、待查询论文摘要词重合总次数
    scoreAbstract4 = 0
    for item in abstract_words:
        scoreAbstract4 += relatedAbstractWords.get(item, 0)
    # -5、待查询论文摘要中词出现次数 
    scoreAbstract5 = 0
    for item in relatedAbstractWords:
        scoreAbstract5 += relatedAbstractWords[item]
        
    
    #两两求比值
    scoreAbstract6 = scoreAbstract3/(scoreAbstract1 + 1)
    scoreAbstract7 = scoreAbstract3/(scoreAbstract2 + 1)
    scoreAbstract8 = scoreAbstract1/(scoreAbstract2 + 1)
    scoreAbstract9 = scoreAbstract4/(scoreAbstract5 + 1)
    scoreAbstract10 = scoreAbstract2/(scoreAbstract4 + 1)
    
    #摘要特征汇总 10
    scoreAbstracts = [scoreAbstract1, scoreAbstract2, scoreAbstract3, scoreAbstract4, scoreAbstract5,
                     scoreAbstract6, scoreAbstract7, scoreAbstract8, scoreAbstract9, scoreAbstract10]
      

    ################ 7、发表时间 ################ 
    # 我们认为同一作者不同年份发表数量均匀
    # 则对偏离平均发表数量的年份做惩罚
    
    scoreYears = []
    
    relatedyears = np.array(list(relatedYear.keys()))
    relatedyears.astype(int)
    relatedyears = relatedyears[relatedyears!=0]
    
    if year == '':
        year = 0
    else:
        year = int(year)
    if year < 1500 or year > 2100:
        year = 0
    
    if (relatedyears.size != 0) & (year != 0):
        # -1、偏离时间惩罚
        scoreYear1 = cleanYear(relatedYear)[year]
        # -2、发表时间与最先发表时间差
        scoreYear2 = np.abs(year - np.min(relatedyears))
        # -3、发表时间与最后发表时间差
        scoreYear3 = np.abs(year - np.max(relatedyears))
        # -4、待查询论文平均发表时间
        scoreYear4 = np.mean(relatedyears)
        # -5、待查询论文发表时间方差
        scoreYear5 = np.std(relatedyears)
        try:
            # -6、该论文发表时间在待查询论文中出现次数
            scoreYear6 = relatedYear[year]
            # -7、该论文发表时间是否出现过
            scoreYear7 = 1
        except:
            scoreYear6,scoreYear7 = 0,0
        #-8、该论文发表时间在待查询论文中出现次数/待查询论文中总发表时间
        scoreYear8 = scoreYear6/sum(relatedYear.values())
        
        #时间特征汇总 10
        scoreYears = [scoreYear1, scoreYear2, scoreYear3, scoreYear4, scoreYear5, scoreYear6, scoreYear7, scoreYear8] 
    else:
        scoreYears = [0.] * 8 


    # 所有特征   13 + 11 + 18 + 9 + 10 + 9 + 8 = 78

    feature = []
    feature += scoreTitles
    feature += scoreAuthors
    feature += scoreVenues
    feature += scoreOrgs
    feature += scoreKeywords
    feature += scoreAbstracts
    feature += scoreYears
    
    return [feature, int(label)]
