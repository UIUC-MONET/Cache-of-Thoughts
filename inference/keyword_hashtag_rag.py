import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

class KeywordExtractor():
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_token_id
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        ]

        self.llama_template = """
        Please give me 10 keywords that are present in this question-option context and separate them with commas.
        Make sure you to only return the keywords and say nothing else.
        Make sure to exclude the words "question" and "options" as keywords
        I have the following multiple choice question and its options:
        - {query}
        """
        # self.message_template = [
        #     {"role": "system", "content": "You are a helpful chatbot that assists users in generating keywords from conversations. You have been given a conversation and need to generate keywords from it."},
        #     {"role": "user", "content": None },
        # ]
    
    def extract_keywords_from_conversation(self, conversation):
        query = 'question: ' + str(conversation['question']) + ' options: ' + str(conversation['options'])
        text = self.apply_keyword_extract_template(query)
        keywords = self.extract_keywords(text)
        return keywords


    def apply_keyword_extract_template(self, sample):
        llama_prompt = self.llama_template.format(
            query=sample
        )
        messages = [
            {"role": "system", "content": "You are a helpful chatbot that assists users in generating keywords from conversations. You have been given a piece of text and need to generate keywords from it."},
            {"role": "user", "content": llama_prompt },
        ]
        return messages


    def extract_keywords(self, text):
        llama_keywords = []
        texts = self.tokenizer.apply_chat_template([text], add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(texts, padding="longest", return_tensors="pt").to(self.device)
        # inputs = {key: val. for key, val in inputs.items()}
        temp_texts=self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)


        gen_tokens = self.model.generate(
            **inputs, 
            max_new_tokens=128, 
            pad_token_id=self.tokenizer.eos_token_id, 
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]

        llama_keywords.extend(gen_text)
        return llama_keywords

    def batch_extract_keywords(self, text_batch):
        llama_keywords = []
        texts = self.tokenizer.apply_chat_template(text_batch, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(texts, padding="longest", return_tensors="pt").to(self.device)
        # inputs = {key: val. for key, val in inputs.items()}
        temp_texts=self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)


        gen_tokens = self.model.generate(
            **inputs, 
            max_new_tokens=128, 
            pad_token_id=self.tokenizer.eos_token_id, 
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]

        llama_keywords.extend(gen_text)
        return llama_keywords

class TextEmbedder:
    def __init__(self, model_name, task_name_to_instruct=None, max_length=32768, use_instruction=True, requires_tokenizer=False, **kwargs):
        self.valid_params = {key: value for key, value in kwargs.items() if value is not None}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) if requires_tokenizer else None
        self.max_length = max_length
        self.task_name_to_instruct = task_name_to_instruct if task_name_to_instruct else {}
        self.use_instruction = use_instruction
        self.requires_tokenizer = requires_tokenizer

    def _get_instruction(self, task):
        if self.use_instruction and task in self.task_name_to_instruct:
            return "Instruct: " + self.task_name_to_instruct[task] + "\nQuery: "
        return ""

    def encode(self, texts, task="default", batch_size=1):
        prefix = self._get_instruction(task)
        texts = [prefix + text for text in texts]  # Add instructions if applicable

        if self.requires_tokenizer and self.tokenizer:  # For models like gte-base-en-v1.5
            # Tokenize input texts
            batch_dict = self.tokenizer(texts, max_length=self.max_length, **self.valid_params)
            batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}  # Move tensors to the correct device

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]  # Get the embeddings (usually the first token)

            # Normalize the embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
        else:  # For models that don’t require tokenizers explicitly
            if batch_size == 1:
                embeddings = self.model.encode(texts, instruction=prefix, max_length=self.max_length)
            else:
                # Mini-batch
                embeddings = self.model._do_encode(
                    texts, batch_size=batch_size, instruction=prefix, max_length=self.max_length, num_workers=4, return_numpy=True
                )
            return F.normalize(torch.tensor(embeddings), p=2, dim=1)

    def similarity_score(self, query_embeddings, passage_embeddings):
        return (query_embeddings @ passage_embeddings.T) * 100


class TextDataset(Dataset):
    def __init__(self, texts, text_prefix=""):
        self.texts = texts
        self.text_prefix = text_prefix

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.text_prefix + self.texts[idx],
        }


class TextEmbedderWithLoader:
    def __init__(self, model_name, task_name_to_instruct=None, max_length=32768):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.max_length = max_length
        self.task_name_to_instruct = task_name_to_instruct if task_name_to_instruct else {}

    def _get_instruction(self, task):
        return "Instruct: " + self.task_name_to_instruct.get(task, "") + "\nQuery: "

    def encode_batch(self, batch, task="default"):
        query_embeddings = self.model.encode(batch["text"], instruction=self._get_instruction(task), max_length=self.max_length)
        return F.normalize(torch.tensor(query_embeddings), p=2, dim=1)

    def similarity_score(self, query_embeddings, passage_embeddings):
        return (query_embeddings @ passage_embeddings.T) * 100


class KeywordEncoder():
    def __init__(self, model_name='Alibaba-NLP/gte-base-en-v1.5'):
        self.model_name = model_name
        self.embedder = TextEmbedder(model_name='Alibaba-NLP/gte-base-en-v1.5', max_length=8192, use_instruction=False, requires_tokenizer=True, padding=True, truncation=True, return_tensors='pt')

        def encode(self, texts):
            embeddings = self.embedder.encode(texts)
            return embeddings