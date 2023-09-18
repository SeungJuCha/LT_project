import os
import openai
import json
import pdb
from LT_project.data.cifar100_classes import Cifar100Class
from tqdm import tqdm

openai.api_key =  #!openai api key input
json_name = "cifar100_description_1.json"

category_list = list(Cifar100Class.keys())
all_responses = {}
# vowel_list = ['A', 'E', 'I', 'O', 'U'] # 영어 단어 시작이 모음인 경우 an으로 시작하기 위해서


def base(k,v):
    moeum = ['a','e','i','o','u']
    if k[0]  in moeum:
        start = 'an'
    else:
        start =  'a'

    if v.endswith(' 1')  or v.endswith(' 2'):
        v = v[:-2]

    middle = start +' ' +k + ', in category of ' + v
    return middle


with open('./LT_project/json/cifar100_class_category.json','r') as f: #! dict of class :coarse label
    class_dict = json.load(f)

for category in tqdm(category_list):

	middle = base(category,class_dict[category])

# 	#! 질문 프롬프트를 생성해보세요??
	prompts = []
	prompts.append("Explain  " + middle +' looks like.')
	prompts.append("What does " + middle+ " look like?")  #! class가 특별히 사람인 경우 no definitive라는 말이 생성되는 경우가 많음 -> failure case
	prompts.append('what are important characteristics of ' + middle + '?')
	prompts.append("What is " + middle+ "?")
	# prompts.append("A caption of an image of "  + article + " "  + category + ":")

	all_result = []
	for curr_prompt in prompts:
		response = openai.Completion.create(
		    engine="text-davinci-002",
		    prompt=curr_prompt,
		    temperature=.99,
			max_tokens = 60,
			n=10,
			stop="." )

		for r in range(len(response["choices"])):
			result = response["choices"][r]["text"]
			all_result.append(result.replace("\n\n", "") + ".")

	all_responses[category] = all_result

with open(json_name, 'w') as f:
	json.dump(all_responses, f, indent=4)

