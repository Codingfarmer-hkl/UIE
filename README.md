# UIE
基于https://github.com/universal-ie/UIE 代码改的，这里仅用于关系抽取
##### [data 数据集](#data数据集)
##### [data_preprocessing 数据预处理代码](#data_preprocessing数据预处理代码)
##### [model_weights 模型文件](#model_weights模型文件)
##### [models 算法代码](#models算法代码)
##### [训练及推理](#训练)
>  ###### [一. 数据预处理](#数据预处理)
>  ######  [二. 微调模型](#微调模型)
>  ###### [三. 模型推理](#模型推理)
##### [数据格式](#1)
> ###### [训练格式input](#2)
>> ###### [解释record](#3)
>> ###### [模型接收格式](#4)
> ###### [推理过程](#5)
>> ###### [训练的实体类型及关系类型](#6)
>> ###### [输入模型的数据格式](#7)
>> ###### [测试输出的结果](#8)
>> ###### [真实的标签](#9)
# <span id="data数据集"> data 数据集</span> 

```
解压data.zip在当前文件夹
raw_data:原始数据集
temp_data:调通模型使用的测试数据集
test_result:测试的结果
train_data:处理后训练数据集
train_data/statistics_train_data:训练数据集的统计信息 
```

# <span id="data_preprocessing数据预处理代码"> data_preprocessing 数据预处理代码</span> 

```
data_preprocessing/data_config:预测配置文件
```

# <span id="model_weights模型文件"> model_weights 模型文件</span> 

```
pre-training-uie-char-small:预训练模型
  从 https://pan.cstcloud.cn/web/share.html?hash=J7HOsDHHQHY 下载模型放在pre-training-uie-char-small目录下
fine-tuned-weights：调优后的模型
```

# <span id="models算法代码"> models 算法代码</span> 

# <span id="训练"> 训练及推理</span> 

## <span id="数据预处理"> 一. 数据预处理</span> 

```
运行 data_preprocessing.py 
```

## <span id="微调模型"> 二. 微调模型</span> 
```
 1. 配置models/config.py文件

 2. 运行 uie_finetune.py
 ```

## <span id="模型推理"> 三. 模型推理</span> 
```
  运行 inference.py
```

# <span id="1"> 数据格式</span> 

## <span id="2"> 训练格式input</span> 

```json
{"text": "笔名：木斧原名：杨莆曾用名：穆新文、牧羊、寒白、洋漾出生日期：1931—职业：作家、诗人性别：男民族：回族政治面貌：中共党员祖籍：固原县出生地：成都", "tokens": ["笔", "名", "：", "木", "斧", "原", "名", "：", "杨", "莆", "曾", "用", "名", "：", "穆", "新", "文", "、", "牧", "羊", "、", "寒", "白", "、", "洋", "漾", "出", "生", "日", "期", "：", "1", "9", "3", "1", "—", "职", "业", "：", "作", "家", "、", "诗", "人", "性", "别", "：", "男", "民", "族", "：", "回", "族", "政", "治", "面", "貌", "：", "中", "共", "党", "员", "祖", "籍", "：", "固", "原", "县", "出", "生", "地", "：", "成", "都"],
"record": "<extra_id_0> <extra_id_0> 人物 <extra_id_5> 木斧 <extra_id_0> 出生日期 <extra_id_5> 1931 <extra_id_1> <extra_id_0> 民族 <extra_id_5> 回族 <extra_id_1> <extra_id_0> 出生地 <extra_id_5> 成都 <extra_id_1> <extra_id_1> <extra_id_0> 日期 <extra_id_5> 1931 <extra_id_1> <extra_id_0> 文本 <extra_id_5> 回族 <extra_id_1> <extra_id_0> 地点 <extra_id_5> 成都 <extra_id_1> <extra_id_1>", 
"entity": [{"type": "人物", "offset": [3, 4], "text": "木斧"}, {"type": "文本", "offset": [51, 52], "text": "回族"}, {"type": "日期", "offset": [31, 32, 33, 34], "text": "1931"}, {"type": "地点", "offset": [72, 73], "text": "成都"}], "relation": [{"type": "民族", "args": [{"type": "人物", "offset": [3, 4], "text": "木斧"}, {"type": "文本", "offset": [51, 52], "text": "回族"}]}, {"type": "出生日期", "args": [{"type": "人物", "offset": [3, 4], "text": "木斧"}, {"type": "日期", "offset": [31, 32, 33, 34], "text": "1931"}]}, {"type": "出生地", "args": [{"type": "人物", "offset": [3, 4], "text": "木斧"}, {"type": "地点", "offset": [72, 73], "text": "成都"}]}], "event": [], "spot": ["地点", "文本", "人物", "日期"], "asoc": ["出生日期", "出生地", "民族"], 
"spot_asoc": [{"span": "木斧", "label": "人物", "asoc": [["出生日期", "1931"], ["民族", "回族"], ["出生地", "成都"]]}, {"span": "1931", "label": "日期", "asoc": []}, {"span": "回族", "label": "文本", "asoc": []}, {"span": "成都", "label": "地点", "asoc": []}]}

```

### <span id="3"> 解释record</span> 

```
sent_start = '<extra_id_0>'
sent_end = '<extra_id_1>'
record_start = '<extra_id_0>'
record_end = '<extra_id_1>'
span_start = '<extra_id_0>'
span_end = '<extra_id_1>'
sep_marker = '<extra_id_2>'
source_span_start = '<extra_id_3>'
source_span_end = '<extra_id_4>'
target_span_start = '<extra_id_5>'
text_start = '<extra_id_2>'
```

### <span id="4"> 模型接收格式</span> 

```
input = input["record"]中的实体类型和关系类型 + text_start + input["text"]
label = record
例如：input = <spot> 人物 <spot> 出生日期<spot><asoc>学校<asoc> + text
```

## <span id="5"> 推理过程</span> 

###  <span id="6"> 输入格式input</span> 

```json
{"text": "蔡志坚在南京艺术学院求学时受过系统、正规的艺术教育和专业训练，深得刘海粟、罗叔子、陈之佛、谢海燕、陈大羽等著名中国画大师的指授，基本功扎实，加上他坚持从生活中汲取创作源泉，用心捕捉生活中最美最感人的瞬间形象，因而他的作品，不论是山水、花鸟、飞禽、走兽，无不充满了生命的灵气，寄托着画家的情怀，颇得自然之真趣"}

```

### <span id="7"> 训练的实体类型及关系类型</span> 

```json
{
    "实体类型":["民族", "出生日期", "出生地", "作曲", "所属专辑", "歌手", "作词", "成立日期", "作者", "连载网站", "毕业院校", "出品公司", "主演", "导演", "出版社", "国籍"],
    "关系类型": ["影视作品", "学校", "国家", "文本", "人物", "地点", "出版社", "音乐专辑", "图书作品", "网站", "日期", "企业", "歌曲", "机构", "网络小说"]
}

```

### <span id="8">输入模型的数据格式</span> 

```
<spot> 人物<spot> 企业<spot> 出版社<spot> 国家<spot> 图书作品<spot> 地点<spot> 学校<spot> 影视作品<spot> 文本<spot> 日期<spot> 机构<spot> 歌曲<spot> 网站<spot> 网络小说<spot> 音乐专辑<asoc> 主演<asoc> 作曲<asoc> 作者<asoc> 作词<asoc> 出品公司<asoc> 出版社<asoc> 出生地<asoc> 出生日期<asoc> 国籍<asoc> 导演<asoc> 成立日期<asoc> 所属专辑<asoc> 歌手<asoc> 毕业院校<asoc> 民族<asoc> 连载网站<extra_id_2>  + input["text"]
```

### <span id="9">测试输出的结果</span> 

```
<extra_id_0><extra_id_0><extra_id_7><extra_id_5>蔡志坚<extra_id_0>出生地<extra_id_5>江苏省南京市<extra_id_1><extra_id_0>毕业院校<extra_id_5>南京艺术学院<extra_id_1><extra_id_0>国籍<extra_id_5>中国<extra_id_1><extra_id_0>出生日期<extra_id_5>1963年<extra_id_1><extra_id_0>生日<extra_id_5>1963年<extra_id_1><extra_id_0>出生地<extra_id_5>南京市<extra_id_1><extra_id_0>民族<extra_id_5>汉族<extra_id_1><extra_id_1><extra_id_0><extra_id_7><extra_id_5>中国画大师<extra_id_0>出品公司<extra_id_5>中国美术出版社<extra_id_1><extra_id_0>出版社<extra_id_5>中国美术出版社<extra_id_1><extra_id_1><extra_id_0><extra_id_7><extra_id_5>中国画<extra_id_0>地点<extra_id_5>中国<extra_id_1><extra_id_1><extra_id_0><extra_id_7><extra_id_5>中国画<extra_id_0>
```

### <span id="10">真实的标签</span> 

```
"<extra_id_0> <extra_id_0> 人物 <extra_id_5> 蔡志坚 <extra_id_0> 毕业院校 <extra_id_5> 南京艺术学院 <extra_id_1> <extra_id_1> <extra_id_0> 学校 <extra_id_5> 南京艺术学院 <extra_id_1> <extra_id_1>"
```








