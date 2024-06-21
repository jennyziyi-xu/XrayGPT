import ast
import csv
import math

def find_value(data, key_to_find):
    for key, value in data:
        if key == key_to_find:
            return value
    return None  # Return None if the key is not found

# read in the outputs.csv file
output_path = "/home/jex451/XrayGPT/outputs/outputs.csv"

with open(output_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

for row in rows[1:]:
    dict_logits = ast.literal_eval(row[0])
    indices = ast.literal_eval(row[1])
    #### Max Prob
    max_prob_para = []
    max_prob_sentence = float("-inf")

    #### Avg Prob 
    avg_prob_para = []
    avg_prob_sentence = []

    for (token_pos, index) in enumerate(indices[1:-2]):
        # get the probability from dict_logits
        # this is softmax probability 
        prob = find_value(dict_logits[token_pos], index)
        # do negative log as shown in the paper. 
        neg_log_prob = -math.log(prob)
        # print("token_pos", token_pos, "prob",neg_log_prob)

        #### Max Prob
        max_prob_sentence = max(max_prob_sentence, neg_log_prob)

        #### Avg Prob
        avg_prob_sentence.append(neg_log_prob)

        # A dot is represented by 29889. 
        if (index == 29889):
            #### Max Prob 
            max_prob_para.append(max_prob_sentence)
            max_prob_sentence = float("-inf")

            #### Avg Prob
            avg_prob_sentence_score = sum(avg_prob_sentence)/len(avg_prob_sentence)
            avg_prob_para.append(avg_prob_sentence_score)

    ### Max Prob
    score_max_prob = sum(max_prob_para) / len(max_prob_para)

    ### Avg Prob 
    score_avg_prob = sum(avg_prob_para) / len(avg_prob_para)

print(avg_prob_para)
print(score_avg_prob)



