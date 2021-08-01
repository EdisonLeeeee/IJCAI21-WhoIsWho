import re
import json
import pickle
from pypinyin import lazy_pinyin

def load_json(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result

def dump_json(fname, obj):
    with open(fname, 'w') as f:
        f.write(json.dumps(obj))
              
def load_pickle(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result
              
def dump_pickle(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)           

def name_remove_comma(name):
    """去除名称中的逗号
    存在如下可能：
    [1] li,_x
    [2] yu,gang
    """
    if ',' in name:
        if '_' in name:
            name = name.replace(',', '')
        else:
            name = name.replace(',', '_')
    return name


def name_remove_zero(name):
    """去除名称中的莫名其妙000
    如：
    ming_li_0006
    """
    name = name.split('_')
    temp = []
    for n in name:
        if not n.startswith('0'):
            temp.append(n)
    return '_'.join(temp)

def ch2en(name):
    """中文名转英文"""
    if not is_Chinese(name): return name
    name = lazy_pinyin(name)
    name = ''.join(name[1:]) + '_' + name[0]
    return name

def name2name(name, reverse=False):
    """三字中文名转成两字
    如：xiao_ming_li -> xiaoming_li
    如果reverse=True，则考虑: li_xiao_ming -> xiaoming_li
    """
    name = name.split('_')
    
    if len(name) == 3 and not length_has_one(name):
        if reverse:
            name = [name[0] + name[1], name[2]]
        else:
            name = [name[1] + name[2], name[0]]
            
    return '_'.join(name)
    
def name_reverse(name):
    """将姓氏和名调换顺序"""
    r = name.split('_')
    if len(r) == 2:
        r = r[::-1]
    return '_'.join(r)

def length_has_one(data):
    for x in data:
        if len(x)==1:
            return True
    return False
    
def find_all_candidate(name, author_data):
    splits = name.split('_')
    template = []
    length_has_one = False
    for n in splits:
        if len(n) == 1:
            template.append(n[0] + '.*')
            length_has_one = True
        else:
            template.append(n)
    if not length_has_one:
        return []            
    if len(template) == 3:
        template = template[0] + template[1] + '_' + template[-1]
    else:
        template = '_'. join(template)

    candidate_name = []
    for n in author_data:
        match = re.match(template, n)
        if match:
            candidate_name.append(n)
    return candidate_name

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

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

mapping_test = {'刘克成': 'kecheng_liu',
 '叶林': 'lin_ye',
 '伍晓春': 'xiaochun_wu',
 '李军': 'jun_li',
 '江华': 'hua_jiang',
 '苏剑': 'jian_su',
 '叶琳': 'lin_ye',
 '汪之国_wang_zhiguo': 'zhiguo_wang',
 '张凌': 'ling_zhang',
 'shucai_li_李术才': 'shucai_li',
 '崔斌': 'bin_cui',
 '周升': 'sheng_zhou',
 '舒桦_shu_hua': 'hua_shu',
 '张红雨': 'hongyu_zhang',
 '黃囇莉': 'lili_huang',
 '周伟': 'wei_zhou',
 '陈伟': 'wei_chen',
 '徐伟': 'wei_xu',
 '张敏_zhang_min': 'min_zhang',
 '李勃': 'bo_li',
 '张敏': 'min_zhang',
 '张民': 'min_zhang',
 '李楠': 'nan_li',
 '徐威': 'wei_xu',
 '周渭': 'wei_zhou',
'wang_yingmin': 'yingmin_wang',
'jian_guo_wang': 'jianguo_wang',
'j_g_wang': 'jianguo_wang',
'xiao_li_li': 'xiaoli_li',
'sun_hongbin': 'hongbin_sun',
'liping_a_cai': 'liping_cai',
'wang_guojun': 'guojun_wang',
'j_z_jiang': 'jianzhong_jiang',
'l_l_huang': 'lili_huang',
'shu_cai_li': 'shucai_li',
'min_zhang_0005': 'min_zhang',
'zhi_qiang_shi': 'zhiqiang_shi',
'li_li_huang': 'lili_huang',
'hong_yu_zhang': 'hongyu_zhang',
'xing_guo_li': 'xingguo_li',
'jian_hong_wu': 'jianhong_wu',
'qiu_yu_zhang': 'qiuyu_zhang',
'li_i_xingguo': 'xingguo_li',
'ott': 'edward_ott',
'hongyu_zhang*': 'hongyu_zhang',
'minzhang': 'min_zhang',
'du_dan_pacific_nw_natl_lab_richland': 'dan_du',
'li_xin_zhang': 'lixin_zhang',
'kobayashi': 'keisuke_kobayashi',    
'yamaguchi': 'akira_yamaguchi',
'xp_dong': 'xiaoping_dong',
'jong_min_lee': 'jongmin_lee',
'l_i_yong':'yong_li',
'masayuki_takahashi': 'masaaki_takahashi'}


mapping_valid = {'qiang_郭强': 'qiang_guo',
'李成': 'cheng_li',
'陈庆': 'qing_chen',
'gang_liu_刘钢': 'gang_liu',
'李勃': 'bo_li',
'刘钢': 'gang_liu',
'张家华': 'jiahua_zhang',
'张明': 'ming_zhang',
'张健': 'jian_zhang',
'张家骅': 'jiahua_zhang',
'陈伟': 'wei_chen',
'邓小明': 'xiaoming_deng',
'余钢': 'gang_yu',
'董林': 'lin_dong',
'陆苇': 'wei_lu',
'张建': 'jian_zhang',
'ying_胡英': 'ying_hu',
'李俊华': 'junhua_li',
'骆广生': 'guangsheng_luo',
'張健': 'jian_zhang',
'王群': 'qun_wang',
'张军': 'jun_zhang',
'ユタカ_ササキ': 'utaka_sasaki',
'alberto_martini':'a_martini',
'a_b_martini':'a_martini',
'martini_alberto':'a_martini',
'andrea_martini':'a_martini',
'alvise_martini':'a_martini',
'martini':'a_martini',
'a_p_martini': 'a_martini',
'andrea_alberto':'a_martini',

'keisuke_l_i_kobayashi': 'keisuke_kobayashi',
'k_w_kobayashi': 'keisuke_kobayashi',
'kobayashi': 'keisuke_kobayashi',    
           
'kumar': 'vijay_kumar',
'vijay_bhooshan_kumar': 'vijay_kumar',
'vijay_k_v_kumar': 'vijay_kumar',        
           
'a_k_sharma': 'ashish_sharma',
'ashish_k_sharma': 'ashish_sharma',
'a_k_sharma': 'ashish_sharma',
'a_p_sharma': 'ashish_sharma',           
'g_s_luo': 'guangsheng_luo',
'gs_luo': 'guangsheng_luo',
'a_m_ding': 'ding_ma',
'm_a_ding': 'ding_ma',
'd_m_ma': 'ding_ma',
'endo': 'akira_endo',
'j_m_liu': 'junming_liu',
'jun‐ming_liu': 'junming_liu',
'da‐wei_wang': 'dawei_wang',
'wei_lü': 'wei_lu',
's_i_hayashi': 'shinichi_hayashi',
'ya_wei_zhang': 'yawei_zhang',
'y_w_zhang': 'yawei_zhang',
'h_y_zhang': 'hongyu_zhang',
'x_m_zhang': 'xuemin_zhang',
'jun_zhang_0003': 'jun_zhang',
'yamaguchi': 'akira_yamaguchi',
'p_m_schuster': 'peter_schuster',
'l_i_ping': 'ping_li',
'wang_shi_ping': 'shiping_wang',
'z_y_yang': 'zhenyu_yang',
'y_u_gang': 'gang_yu',
's_p_wang': 'shiping_wang',
'c_s_wang': 'chunsheng_wang',
'xiaoming_m_deng': 'xiaoming_deng',
}