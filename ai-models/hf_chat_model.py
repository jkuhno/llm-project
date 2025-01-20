
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


class ChatModel:
    def __init__(self, model_name, quantization_config):
        self.model_name = model_name
        self.quantization_config = quantization_config


    def get_model(self):
        llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=self.quantization_config
            )

            # To support langchain, a pipeline is a must (?)
            # With 8bit quantization, a pipeline is reported to cause slowdown. Needs to be optimized
        pipe = pipeline(
                "text-generation",
                model=llm_model,
                tokenizer=llm_tokenizer,
                max_new_tokens=50,
                return_full_text=False
        )
        hf_pipe = HuggingFacePipeline(pipeline=pipe)
        chat_model = ChatHuggingFace(llm=hf_pipe)

        return chat_model