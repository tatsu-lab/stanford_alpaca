
import os
import openai

#openai.api_key = 'sk-proj-1j99oXeEQvOxtnJXJMbXT3BlbkFJvQ8bykzxjJSNfXIBszux'
#os.environ['OPENAI_API_KEY'] = 'sk-proj-1j99oXeEQvOxtnJXJMbXT3BlbkFJvQ8bykzxjJSNfXIBszux'

openai.api_key = 'sk-proj-QnqAfveIAngLtnrWRpSDT3BlbkFJV6LGRSseZXbnw9AytSoH'
os.environ['OPENAI_API_KEY'] = 'sk-proj-QnqAfveIAngLtnrWRpSDT3BlbkFJV6LGRSseZXbnw9AytSoH'

import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils
import ollama

import fire

def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt

def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        blacklist = [
            "billede", "billeder", "graf", "grafer", "billede", "billeder", "fil", "filer", "kort", "kort", "tegne", 
            "diagram", "gå til", "video", "lyd", "musik", "flowchart", "diagram",
        ]   
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        if inst.startswith("Write a program"):
            continue
        if inst[0] in string.punctuation:
            continue
        if not inst[0].isascii():
            continue
        # Ensure instruction is in Danish (rudimentary check for Danish words)
        if not any(word in inst.lower() for word in ["hvordan", "hvad", "hvorfor", "forklar", "beskriv"]):
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions



def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./MikkelTEST",
    seed_tasks_path="./new_seed_tasks.jsonl",
    num_instructions_to_generate=3,
    model_name="gpt-3.5-turbo-instruct",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
            
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )
        request_start = time.time()

        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))

"""## the data"""



generate_instruction_following_data(
    output_dir="./AlleInstructionsFinal",
    seed_tasks_path="./seed_tasks_dansk.jsonl",
    num_instructions_to_generate=10000,
    model_name="gpt-3.5-turbo-instruct",
    num_prompt_instructions=3,
    request_batch_size=2,
    temperature=1.0,
    top_p=1.0,
    num_cpus=6,
)




# import json

# file_path = "new_tasks_Danish45/regen.json"

# with open(file_path, 'r') as json_file:
#     data = json.load(json_file)

# # drop the most similar as measured by Rouge
# for dictionary in data:
#     dictionary.pop('most_similar_instructions', None)
#     dictionary.pop('avg_similarity_score',None)


# print(data)

# len(data)

# data[16]

# data[3]

# data[5]






# import json

# file_path = "alpaca_data.json"

# with open(file_path, 'r') as json_file:
#     data = json.load(json_file)

# # print(data)
# data[12]

# """## Seed Tasks"""

# seed_tasks_path="seed_tasks.jsonl"

# seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]

# seed_instruction_data = [
#     {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
#     for t in seed_tasks
# ]
# print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

# seed_tasks[12]

# """## Creating a new set of seed tasks"""

# new_seed_tasks = seed_tasks[:3]

# new_seed_tasks


# json_new_tasks = '''
# [
#     {
#         "id": "new_seed_task_3",
#         "name": "vegetariske_middagsmuligheder",
#         "instruction": "Hvad er nogle vegetariske middagsmuligheder med højt proteinindhold?",
#         "instances": [{"input": "",
#                       "output": "Nogle vegetariske middagsmuligheder med højt proteinindhold inkluderer linsesuppe, kikærtekarry, tofu-stir-fry, tempeh-tacos og quinoa-fyldte peberfrugter."}],
#         "is_classification": false
#     },
#     {
#         "id": "new_seed_task_4",
#         "name": "forklaring_af_refundering_politik",
#         "instruction": "Kan du forklare din refunderingspolitik?",
#         "instances": [{"input": "",
#                       "output": "Vores refunderingspolitik tillader kunder at anmode om en refundering inden for 30 dage efter købet, hvis de er utilfredse med produktet eller tjenesten. For at være berettiget til en refundering skal produktet være i sin originale stand og emballage, og tjenester må ikke være afsluttet."}],
#         "is_classification": false
#     },
#     {
#         "id": "new_seed_task_5",
#         "name": "fejlfinding_wifi_forbindelse",
#         "instruction": "Jeg har problemer med at oprette forbindelse til mit Wi-Fi-netværk. Hvad skal jeg gøre?",
#         "instances": [{"input": "",
#                       "output": "Først skal du tjekke, om din enhed er inden for rækkevidde af din Wi-Fi-router. Bekræft derefter, at du har indtastet det korrekte netværksnavn og adgangskode. Hvis problemet fortsætter, skal du prøve at genstarte både din enhed og routeren. Hvis du stadig ikke kan oprette forbindelse, skal du kontakte din internetudbyder for yderligere assistance."}],
#         "is_classification": false
#     },
#     {
#         "id": "new_seed_task_6",
#         "name": "ændre_adgangskode_instruktioner",
#         "instruction": "Hvordan ændrer jeg min adgangskode?",
#         "instances": [{"input": "",
#                       "output": "For at ændre din adgangskode, skal du logge ind på din konto, gå til indstillingerne eller kontoadministrationssektionen og finde muligheden for at ændre din adgangskode. Indtast din nuværende adgangskode, og indtast derefter din nye adgangskode to gange for bekræftelse. Gem dine ændringer for at opdatere din adgangskode."}],
#         "is_classification": false
#     },
#     {
#         "id": "new_seed_task_7",
#         "name": "forespørgsel_om_mistet_pakke",
#         "instruction": "Min pakke er ikke ankommet endnu. Hvad skal jeg gøre?",
#         "instances": [{"input": "",
#                       "output": "Først skal du tjekke din ordrebekræftelsesmail eller konto for sporingsnummeret og leveringsdatoen. Hvis leveringsdatoen er overskredet, skal du kontakte fragtfirmaet med dit sporingsnummer for en opdatering. Hvis fragtfirmaet ikke kan løse problemet, bedes du kontakte vores kundesupport for assistance."}],
#         "is_classification": false
#     },
#     {
#         "id": "new_seed_task_8",
#         "name": "ændre_ordren",
#         "instruction": "Jeg har brug for at ændre min ordre. Hvordan kan jeg gøre det?",
#         "instances": [{"input": "",
#                       "output": "For at ændre din ordre bedes du kontakte vores kundesupport så hurtigt som muligt med dit ordrenummer og de ændringer, du ønsker at foretage. Bemærk, at ændringer måske ikke er mulige, hvis ordren allerede er behandlet eller afsendt."}],
#         "is_classification": false
#     }
# ]
# '''


# new_seed_tasks = json.loads(json_new_tasks)
# print(new_seed_tasks)

# with open('new_seed_tasks.jsonl', 'w') as outfile:
#     for task_dict in new_seed_tasks:
#         json.dump(task_dict, outfile)
#         outfile.write('\n')

