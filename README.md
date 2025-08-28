[原项目](https://github.com/OpenStellarTeam/ChineseSimpleQA)

对原项目的python脚本`judge/chinese_simpleqa_easy_alter_judge.py`进行了修改和优化，标注了几处需要修改参数的地方，同时修复了判断模型未找到的问题（脚本中固定为了gpt-4o-0806），添加了自定义判断模型的函数（可修改）

添加了`make_csv.py`，用于在`judge/chinese_simpleqa_easy_alter_judge.py`无法输出时，手动查看并删除测试得到的jsonl中的错误写入的数据（通常表现为被截断的jsonl条目`{}`）后，手动输出csv文件

~~拙劣的修改~~

项目用于课题使用，与上游项目一致采用MIT协议
