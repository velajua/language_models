import os
import dill

from datetime import datetime
from google.cloud import storage

from transformers import AutoModelForSeq2SeqLM
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification

FILE_PREF: str = '' if 'language_models' in os.getcwd() else '/tmp/'
today = datetime.now().strftime('%Y_%m_%d')


def upload_file_blob(local_path, cs_path, bucket_name):
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(cs_path)
    blob.upload_from_filename(local_path)


def upload_model(model_name, today):
    upload_file_blob(f'{model_name}',
                     f"historic/{today}_{model_name}",
                     'my_model_deployment')
    upload_file_blob(f'{model_name}',
                     f"{model_name}",
                     'my_model_deployment')


def deploy_models():

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    # https://huggingface.co/dslim/bert-base-NER


    class ModelWrapper():

        def __init__(self, model, tokenizer):
            from transformers import pipeline
            self.model = pipeline(
                "ner", model=model,
                tokenizer=tokenizer)

        def prepare_data(self, input_data):
            return input_data.strip()

        def predict(self, input_data):
            rsp = {'statusCode': 200}
            adjusted_x = self.prepare_data(input_data)
            pred = self.model(adjusted_x)  
            rsp["body"] = pred
            return rsp
        
    mw = ModelWrapper(model, tokenizer)

    with open((model_name := f"{FILE_PREF}entity_match_model.dill"), "wb") as f:
        dill.dump(mw, f)

    upload_model(model_name, today)

    tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-ru")
    model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-ru")
    # https://huggingface.co/facebook/wmt19-en-ru


    class ModelWrapper():

        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def prepare_data(self, input_data):
            return self.tokenizer.encode(input_data.strip(),
                                    return_tensors="pt")

        def predict(self, input_data):
            rsp = {'statusCode': 200, "body": []}
            adjusted_x = self.prepare_data(input_data)
            pred = self.tokenizer.decode(
                self.model.generate(adjusted_x)[0],
                skip_special_tokens=True)
            rsp["body"] = pred
            return rsp

    mw = ModelWrapper(model, tokenizer)

    with open((model_name := f"{FILE_PREF}russian_translation_model.dill"), "wb") as f:
        dill.dump(mw, f)

    upload_model(model_name, today)

    tokenizer = AutoTokenizer.from_pretrained("snrspeaks/t5-one-line-summary")
    model = AutoModelForSeq2SeqLM.from_pretrained("snrspeaks/t5-one-line-summary")
    # https://huggingface.co/snrspeaks/t5-one-line-summary


    class ModelWrapper():

        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def prepare_data(self, input_data):
            return self.model.generate(
                input_ids=self.tokenizer.encode(
                    "summarize: " + input_data,
                    return_tensors="pt",
                    add_special_tokens=True),
                num_beams=5, max_length=50,
                repetition_penalty=2.5, length_penalty=1,
                early_stopping=True, num_return_sequences=3)

        def predict(self, input_data):
            rsp = {'statusCode': 200, "body": []}
            adjusted_x = self.prepare_data(input_data)
            pred = [self.tokenizer.decode(
                g, skip_special_tokens=True,
                clean_up_tokenization_spaces=True)
                    for g in adjusted_x]
            rsp["body"] = pred
            return rsp

    mw = ModelWrapper(model, tokenizer)

    with open((model_name := f"{FILE_PREF}summarize_data_model.dill"), "wb") as f:
        dill.dump(mw, f)

    upload_model(model_name, today)
