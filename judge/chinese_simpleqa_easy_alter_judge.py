import requests
import json
import traceback
import re
import sys
from time import sleep
from tqdm.asyncio import tqdm
from multiprocessing import Pool
import numpy as np
from openai import OpenAI
import pandas as pd
import os
os.environ["OPENAI_API_KEY"] = "your key"   #put your own test model name here! 把你的测试模型key写在这里！
os.environ["OPENAI_BASE_URL"] = "your url"   #put your own test model url here! 把你的测试模型api端点写在这里！

judge_prompt = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

以下是【未尝试】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
    - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
    - 预测答案“3520”和“3600”均为【错误】。 
    - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题“菱镁矿的主要化学成分是什么？”标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {target}
预测答案: {predicted_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()

judge_prompt_en = """\
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

def call_model(messages, modelname):
    k = 3
    ouput = ""
    while(k > 0):
        k -= 1
        try:
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_BASE_URL"],
            )
            completion = client.chat.completions.create(
                model=modelname,
                messages=messages,
                stream=False,
                temperature=0.8
            )
            ouput = completion.choices[0].message.content
            if ouput != None and ouput != "":
                break
        except Exception as e:
            print(e)
            time.sleep(10)
            print("Retrying...")
            continue
    return ouput, None

def call_model_judge(messages, modelname):
    k = 3
    ouput = ""
    while(k > 0):
        k -= 1
        try:
            client = OpenAI(
                api_key="judge api key",   #put your own judge model key here! 把你的判断模型key写在这里！
                base_url="judge base url",   #put your own judge model base url here! 把你的判断模型api端点写在这里！
            )
            completion = client.chat.completions.create(
                model=modelname,
                messages=messages,
                stream=False,
                temperature=0.8
            )
            ouput = completion.choices[0].message.content
            if ouput != None and ouput != "":
                break
        except Exception as e:
            print(e)
            continue
    return ouput, None

def write_to_file(info):
    if not isinstance(info, str):
        info = json.dumps(info, ensure_ascii=False)
    with open(new_file, 'a', encoding='utf-8') as fin:
        fin.write(info + '\n')


def judge_answer(question, ref_answer, answer):
    prompt = judge_prompt.format(question = question, target = ref_answer, predicted_answer = answer)
    messages = [{"role": "system", "content": "你是一个智能助手，请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。"}]
    messages.append({"role": "user", "content": prompt})
    output,_ = call_model_judge(messages, "judge model")   #put your own judge model name here! 把你的判断模型名写在这里！
    correct = "C"
    try:
        match = re.search(r"(A|B|C)", output)
        correct = match.group(0) if match else "C" 
    except:
        output = "error"
    judge = {"judge": output.strip(), "score": correct}
    return judge

def process_line(line):
    score = line.get("score", "")
    if score != "":
        write_to_file(line)
        return 1
    
    question = line['question']
    ref_answer = line['answer']
    messages = [{"role": "system", "content": "你是一个智能助手。"}]
    messages.append({"role": "user", "content": question})
    try:
        output = line.get("model_output", "")
        if output == "":
          output,_ = call_model(messages, call_modelname)
        print("\noutput: ", output)    
        line['model_output'] = output
        if output == "":
            line['info'] = "模型输出为空"
            write_to_file(line)
            return 0
        judge = judge_answer(question, ref_answer, output)
        if judge['judge'] == "" or judge['judge'] == "error":
            line['info'] = "评判出错"
            write_to_file(line)
            return 0
        line['judge'] = judge
        line['score'] = judge['score']
        write_to_file(line)
        return 1
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        write_to_file(line)
        return 0

def calculate_accuracies(group):
    total_questions = len(group)
    total_correct = group[group['score'] == "A"].shape[0]
    total_incorrect = group[group['score'] == "B"].shape[0]
    total_not_attempted = group[group['score'] == "C"].shape[0]
    
    total_correct_accuracy = total_correct / total_questions if total_questions > 0 else 0
    total_incorrect_accuracy = total_incorrect / total_questions if total_questions > 0 else 0
    total_not_attempted_accuracy = total_not_attempted / total_questions if total_questions > 0 else 0
    
    total_given_attempted_accuracy = total_correct / (total_correct + total_incorrect) if (total_correct + total_incorrect) > 0 else 0
    
    f1 = 2 * total_given_attempted_accuracy * total_correct_accuracy / (total_given_attempted_accuracy+ total_correct_accuracy) if (total_given_attempted_accuracy+ total_correct_accuracy) > 0 else 0

    return pd.Series({'correct': total_correct_accuracy, 'incorrect': total_incorrect_accuracy, 'not_attempted': total_not_attempted_accuracy, "given_attempted_accuracy": total_given_attempted_accuracy, "F1": f1})




# ==============================
print("Usage: python judge/chinese_simpleqa_easy.py <call_modelname>")

output_name=input("output_name(ex.qwen/qwen3-32b)=")

print(f"output file name = {folder}/simpleqa_{output_name}_!t=0.8.jsonl")

call_modelname = sys.argv[1]

print(f"Model Name: {call_modelname}")

origin_file = "data/chinese_simpleqa.jsonl"
folder = "evaluation/models_results"
if not os.path.exists(folder):
    os.makedirs(folder)
new_file = f"{folder}/simpleqa_{output_name}_!t=0.8.jsonl"



# ==============================
      
if __name__ == "__main__":
    import time
    start_time = time.perf_counter()   
    done = {} 
    if os.path.exists(new_file):
        with open(new_file, "r", encoding='utf-8') as fin:
            done_lines = fin.readlines()
            for line in done_lines:
                data = json.loads(line)
                model_output = data.get("model_output", "")
                if model_output != "":
                    question = data['question']
                    done[question] = data
    data_new = []
    with open(origin_file, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            data = json.loads(line)
            question = data['question']
            if question in done:
                data_new.append(done[question])
            else:
                data_new.append(data)
    fin =  open(new_file, "w", encoding='utf-8')
    with Pool(processes=10) as pool:  #change the process here to change parallel amount! 改变这里的process数可以改变并行数量
        sleep(1)
        results = list(tqdm(pool.imap(process_line, data_new), total=len(lines)))
        correct = np.sum(np.array(results))
        print("成功数： ", correct)
    k = 0 
    
    while correct < int(len(data_new)*0.95) and k < 3:
        k += 1
        print("失败数量过多，重试")
        start_time = time.perf_counter()    
        origin_file = new_file
        with open(origin_file, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
            lines = [json.loads(line) for line in lines]
        new_file = f"{new_file}_{k}.jsonl"
        fin =  open(new_file, "w", encoding='utf-8')
        with Pool(processes=1) as pool:    #dont change the process here! 不要改变这里的process数！
            results = list(tqdm(pool.imap(process_line, lines), total=len(lines)))       
            correct = np.sum(np.array(results))
            print("成功数： ", correct)   
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) / 60
    print(f"执行耗时: {execution_time_ms} mins")
    with open(new_file, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
        datas = [json.loads(line) for line in lines]
        df = pd.json_normalize(datas)
        accuracy_df = calculate_accuracies(df)

        overall_row = pd.DataFrame({
            'primary_category': ['Overall'],
            'correct': [accuracy_df['correct']],
            'incorrect': [accuracy_df['incorrect']],
            'not_attempted': [accuracy_df['not_attempted']],
            'given_attempted_accuracy': [accuracy_df['given_attempted_accuracy']],
            'F1': [accuracy_df['F1']],
        })

        accuracy_df = df.groupby('primary_category').apply(calculate_accuracies).reset_index()
        final_df = pd.concat([overall_row, accuracy_df], ignore_index=True)
        final_df.to_csv(new_file.replace(".jsonl", ".csv"), index=False)
