import re
import random
from base import Agent
from colorama import Fore, Style
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
from transformers import logging as transformers_logging

from utils import RAG, AdaptiveRAG, strip_all_lines

# Ignore warning messages from transformers
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

class ClassificationAgent(Agent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    @staticmethod
    def get_system_prompt() -> str:
        '''
        system_prompt = """\
        Act as a professional medical doctor that can diagnose the patient based on the patient profile.
        Provide your diagnosis in the following format: <number>. <diagnosis>""".strip()
        '''
        
        system_prompt = """\
        Act as a professional medical doctor that can diagnose the patient based on the patient profile.
        Provide your reasoning process step-by-step, and then provide your diagnosis in the following format:
        Reasoning: <step-by-step reasoning>
        Diagnosis: <number>. <diagnosis>""".strip()
        
        return strip_all_lines(system_prompt)

    @staticmethod
    def get_zeroshot_prompt(option_text: str, text: str) -> str:
        '''
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the following patient profile:
        {text}

        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Now, directly provide the diagnosis for the patient in the following format: <number>. <diagnosis>""".strip()
        '''
        
        prompt = f"""\ 
        Act as a medical doctor and diagnose the patient based on the following patient profile:
        {text}
        
        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}
        
        Now provide your reasoning process step-by-step, and then provide your diagnosis in the following format:
        Reasoning: <step-by-step reasoning>
        Diagnosis: <number>. <diagnosis>""".strip()
        
        return strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        {{question}}
        Diagnosis: {{answer}}"""
        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(option_text: str, text: str,) -> str:
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the provided patient profile.
        
        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}

        Here are some example cases.
        
        {{fewshot_text}}
        
        Now it's your turn.
        
        {text}        
        
        Now provide the diagnosis for the patient in the following format: <number>. <diagnosis>"""
        
        return strip_all_lines(prompt)
    
    def generate_response(self, messages: list) -> str:
        """
        Generate a response using the local model.
        """
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        #return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        reasoning = re.search(r"Reasoning:(.*?)(?:Diagnosis:|$)", response_text, re.S)
        diagnosis = re.search(r"Diagnosis:(.*?)(?:$)", response_text, re.S)
        return {
            "reasoning": reasoning.group(1).strip() if reasoning else "",
            "diagnosis": diagnosis.group(1).strip() if diagnosis else ""
        }
    
    '''
    @staticmethod
    def extract_label(pred_text: str, label2desc: dict[str, str]) -> str:
        numbers = re.findall(pattern=r"(\d+)\.", string=pred_text)
        
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the first one." + Style.RESET_ALL)
                prediction = numbers[0]
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        return str(prediction)
    '''

    @staticmethod
    def extract_label(response: dict, label2desc: dict[str, str]) -> tuple:
        """
        Extract diagnosis and reasoning from the response.
        """
        pred_text = response.get("diagnosis", "")
        reasoning = response.get("reasoning", "")

        # Extract diagnosis number
        numbers = re.findall(r"(\d+)\.", pred_text)
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the first one." + Style.RESET_ALL)
                prediction = numbers[0]
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))

        return prediction, reasoning

    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        
        # TODO
        super().__init__(config)
        self.llm_config = config
        
        if config['use_8bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                quantization_config=quantization_config,
                device_map=config["device"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map=config["device"]
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        
        '''
        self.rag = RAG(config["rag"])
        '''
        self.rag = AdaptiveRAG(config["rag"])
        
        self.reasoning_logs = []
        
        # Save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.self_outputs = list()
        
        self.model.eval()

    def __call__(self, label2desc: dict[str, str], text: str) -> str:
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.

        For example:
        label2desc = {
            "apple": "A fruit that is typically red, green, or yellow.",
            "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
            "cherry": "A small, round stone fruit that is typically bright or dark red.",
        }
        text = "The fruit is red and about the size of a tennis ball."
        label = "apple" (should be a key in label2desc, i.e., ["apple", "banana", "cherry"])
        """
        
        # TODO
        self.reset_log_info()
        option_text = '\n'.join([f"{str(k)}. {v}" for k, v in label2desc.items()])
        system_prompt = self.get_system_prompt()
        prompt_zeroshot = self.get_zeroshot_prompt(option_text, text)
        prompt_fewshot = self.get_fewshot_template(option_text, text)
        
        '''
        shots = self.rag.retrieve(query=text, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
        '''
        
        retrieval_results = self.rag.retrieve(query=text)
        docs, scores = zip(*retrieval_results) if retrieval_results else ([], [])

        weights = self.rag.adjust_weights(scores)
        shots = [f"[Weight: {weight:.2f}] {doc}" for doc, weight in zip(docs, weights)]

        if self.rag.insert_acc >= 150:
            if len(shots) > 0:
                fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
                try:
                    prompt = re.sub(pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot)
                except Exception as e:
                    error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                    print(Fore.RED + error_msg + Fore.RESET)
                    prompt = prompt_zeroshot
            else:
                print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
                prompt = prompt_zeroshot
        else:
            prompt = prompt_zeroshot     

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.generate_response(messages)
        prediction, reasoning = self.extract_label(response, label2desc)
        
        '''
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(system_prompt + prompt)),
            "num_output_tokens": len(self.tokenizer.encode(response)),
            "num_shots": str(len(shots)),
            "input_pred": prompt,
            "output_pred": response,
        })
        '''
        self.reasoning_logs.append({
            "input": text,
            "reasoning": reasoning,
            "diagnosis": prediction
        })
        self.inputs.append(text)
        self.self_outputs.append(f"{str(prediction)}. {label2desc[int(prediction)]}")
        
        return prediction
    
    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own prediction at the current time step.

        Args:
            correctness (bool): Whether the prediction is correct.

        Returns:
            bool: Whether the prediction is correct.
        """
        
        # TODO
        if correctness:
            '''
            question = self.inputs[-1]
            answer = self.self_outputs[-1]
            chunk = f"Patient Profile: {question}\nReasoning: {reasoning}\nDiagnosis: {answer}"
            self.rag.insert(key=question, value=chunk)
            '''
            
            current_log = self.reasoning_logs[-1]  # 取最後一條記錄
            question = current_log["input"]
            reasoning = current_log["reasoning"]
            answer = current_log["diagnosis"]
            
            chunk = f"Patient Profile: {question}\nReasoning: {reasoning}\nDiagnosis: {answer}"
            self.rag.insert(key=question, value=chunk)
            
            '''
            if self.rag.insert_acc % 50 == 0:
                self.rag.update_memory(top_k=500)
            '''
            
            return True
        
        return False

class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        raise NotImplementedError

    def __call__(
        self,
        table_schema: str,
        user_query: str
    ) -> str:
        """
        Generate SQL code based on the given table schema and the user query.

        Args:
            table_schema (str): The table schema.
            user_query (str): The user query.

        Returns:
            str: The SQL code that the LLM generates.
        """
        # TODO: Note that your output should be a valid SQL code only.
        raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own SQL    code at the current time step.
        """
        # TODO
        raise NotImplementedError
        
if __name__ == "__main__":
    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--output_path', type=str, default=None, help='path to save csv file for kaggle submission')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        max_tokens = 16
        agent_name = ClassificationAgent
    elif args.bench_name.startswith("sql_generation"):
        max_tokens = 512
        agent_name = SQLGenerationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")

    bench_cfg = {
        'bench_name': args.bench_name,
        'output_path': args.output_path
    }
    llm_config = {
        # TODO: specify your configs for the agent here
        'model_name': args.model_name,
        'exp_name': f'adaptive_rag_{args.bench_name}_{args.model_name}',
        'bench_name': bench_cfg['bench_name'],
        'max_tokens': max_tokens,
        'do_sample': False,
        'device': args.device,
        'use_8bit': args.use_8bit,
        'rag': {
            #'embedding_model': 'BAAI/bge-base-en-v1.5',
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            #'embedding_model': 'medicalai/ClinicalBERT',
            #'embedding_model': 'emilyalsentzer/Bio_ClinicalBERT',
            #'embedding_model': 'NeuML/pubmedbert-base-embeddings',
            #'embedding_model': 'pritamdeka/S-PubMedBert-MS-MARCO',
            #'embedding_model': 'abhinand/MedEmbed-large-v0.1',
            'seed': 42,
            'top_k': 16,
            'order': 'similar_at_top',
            'embed_dim': 768,
        }
    }
    agent = agent_name(llm_config)
    main(agent, bench_cfg, debug=args.debug, use_wandb=args.use_wandb, wandb_name=llm_config["exp_name"], wandb_config=llm_config)
