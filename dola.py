import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, LLamaQaStoppingCriteria
from transformers.generation.configuration_utils import GenerationConfig

class DoLa:
    def __init__(self, model_name, device, num_gpus, dola_layers, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory
        self.dola_layers = dola_layers
        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    # load_in_8bit=True, # LoRA
                                                    #load_in_4bit=True, # Quantization Load
                                                    # torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True,
                                                     **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode(stop_word)
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, **kwargs):
        with torch.no_grad():
            dola_config = GenerationConfig(do_sample=False, max_new_tokens=30, dola_layers=self.dola_layers,
                                           early_stopping = True,
                                           repetition_penalty = 1.2,
                                          pad_token_id = self.tokenizer.pad_token_id,
                                          eos_token_id = self.tokenizer.eos_token_id,
                                          return_dict_in_generate=True,
                                          output_scores=True,
                                          output_logits = True)
            config = GenerationConfig(do_sample=False, max_new_tokens=30,
                                        early_stopping = True,
                                          pad_token_id = self.tokenizer.pad_token_id,
                                          eos_token_id = self.tokenizer.eos_token_id,
                                          return_dict_in_generate=True,
                                          output_scores=True,
                                          output_logits = True)                              
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            origin_outputs = self.model.generate(input_ids, generation_config = config, **kwargs)
            dola_outputs = self.model.generate(input_ids, generation_config = dola_config, **kwargs)

            # skip the tokens in the input prompt
            sequences, scores, logits = dola_outputs.sequences, dola_outputs.scores, dola_outputs.logits
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            origin_sequences = origin_outputs.sequences
            origin_sequences = origin_sequences[:, input_ids.shape[-1]:][0, :]
            origin_output_str = self.tokenizer.decode(origin_sequences, skip_special_tokens=True)

        if self.device:
            torch.cuda.empty_cache()

        return output_str, origin_output_str
