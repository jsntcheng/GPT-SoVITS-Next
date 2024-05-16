'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os, re, logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import LangSegment
import torch
import librosa
import gradio as gr
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from loguru import logger as log
from time import time as ttime
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.path.exists("./gweight.txt"):
    with open("./gweight.txt", 'r', encoding="utf-8") as file:
        gweight_data = file.read()
        gpt_path = os.environ.get(
            "gpt_path", gweight_data)
else:
    gpt_path = os.environ.get(
        "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

if os.path.exists("./sweight.txt"):
    with open("./sweight.txt", 'r', encoding="utf-8") as file:
        sweight_data = file.read()
        sovits_path = os.environ.get("sovits_path", sweight_data)
else:
    sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

i18n = I18nAuto()
class ModelException(Exception):
    pass

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


# 开始重构
class BaseTorchModel:
    def __init__(self,model_path,device):
        self.model = None
        self.model_is_init = False 
        self.model_is_load = False
        self.alow_sleep = False
        self.model_awake = False
        self.name = '基础'
        self.model_path = model_path
        if device == 'cuda':
            if self.check_cuda():
                pass
            else:
                log.warning('无可用显卡，改为cpu')
                device = 'cpu'
        self.device = device
                
    def check_cuda(self):
        return torch.cuda.is_available()
    
    def load_model(self):
        if not self.model_is_init:
            raise ModelException('模型未初始化')
        self.is_half = False
        self.dtype = torch.float32
        if self.device != 'cuda':
            self.model = self.model.to(self.device)
        
        else:
            try:
                self.model = self.model.half().to(device)
                self.is_half = True
                self.dtype = torch.float16
            except:
                log.warning('显卡不支持半精度，改为全精度加载')
                self.model = self.model.to(self.device)
        self.model_awake = True
        self.model_is_load = True
    
    def sleep_model(self,device=None):
        if self.alow_sleep:
            if device == None:
                device = self.device
            if device == 'cuda':
                self.model.cpu()
                self.model.eval()
                self.model_awake = False
            log.info(f'{self.name}模型已休眠')
        torch.cuda.empty_cache()
    
    def awake_model(self,device = None):
        if self.alow_sleep:
            if device == None:
                device = self.device
            if self.model_awake == False and device == 'cuda':
                if self.is_half:
                    self.model.half().cuda()
                else:
                    self.model.cuda()
                self.model.eval()
                self.model_awake = True
            else:
                self.model_awake = True
            log.info(f'{self.name}模型已唤醒')
        torch.cuda.empty_cache()
        
class BertModel(BaseTorchModel):
    def __init__(self,model_path,device):
        super().__init__(model_path,device)
        self.name = 'Bert'
        self.init_tokenizer()
        self.init_model()
        
    def init_model(self):
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_path)
        self.model_is_init = True
        
    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    
    @staticmethod
    def clean_text_inf(text, language):
        phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text
    
    def get_bert_feature(self,text, word2ph, device=None):
        if device is None:
            device = self.device
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = self.model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T
    
    def get_bert_inf(self, phones, word2ph, norm_text, language, device=None):
        if device is None:
            device = self.device
        if language.replace("all_","") == "zh":
            bert = self.get_bert_feature(norm_text, word2ph, device).to(self.dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=self.dtype
            )

        return bert.to(device)

    def get_phones_and_bert(self,text,language,use_device=None,output_device=None):
        if output_device is None:
            if use_device is None:
                output_device = self.device
            else:
                output_device = use_device
        if use_device is None:
            use_device = self.device
        
        self.awake_model(use_device)
            
        if language in {"en","all_zh","all_ja"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
            if language == "zh":
                bert = self.get_bert_feature(norm_text, word2ph)
            else:
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=self.dtype,
                ).to(use_device)
        elif language in {"zh", "ja","auto"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en","ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "ko":
                        langlist.append("zh")
                        textlist.append(tmp["text"])
                    else:
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang, use_device)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)
        self.sleep_model()
        return phones,bert.to(output_device).to(self.dtype),norm_text

class CnhubertModel(BaseTorchModel):
    def __init__(self,model_path,device):
        super().__init__(model_path,device)
        self.name = 'SSL'
        self.init_model()
    
    def init_model(self):
        cnhubert.cnhubert_base_path = self.model_path
        self.model = cnhubert.get_model()
        self.model_is_init = True

class SovitsModel(BaseTorchModel):
    def __init__(self,model_path,device):
        super().__init__(model_path,device)
        self.name = 'Sovits'
        self.init_model()
    
    def load_model(self):
        if not self.model_is_init:
            raise ModelException('模型未初始化')
        self.is_half = False
        self.dtype = torch.float32
        if self.device != 'cuda':
            self.model = self.model.to(self.device)
        
        else:
            try:
                self.model = self.model.half().to(device)
                self.is_half = True
                self.dtype = torch.float16
            except:
                log.warning('显卡不支持半精度，改为全精度加载')
                self.model = self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(self.dict_s2["weight"], strict=False)
        self.model_awake = True
        self.model_is_load = True
    
    
    def init_model(self):
        self.dict_s2 = torch.load(self.model_path, map_location="cpu")
        self.hps = DictToAttrRecursive(self.dict_s2["config"])
        self.hps.model.semantic_frame_rate = "25hz"
        self.model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        if ("pretrained" not in sovits_path):
            del self.model.enc_q
        self.model_is_init = True

    def get_spepc(self,filename):
        audio = load_audio(filename, int(self.hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        del audio_norm,audio
        torch.cuda.empty_cache()
        return spec

class GptModel(BaseTorchModel):
    def __init__(self,model_path,device):
        super().__init__(model_path,device)
        self.name = 'GPT'
        self.init_model()

    def load_model(self):
        if not self.model_is_init:
            raise ModelException('模型未初始化')
        self.is_half = False
        self.dtype = torch.float32
        if self.device != 'cuda':
            self.model = self.model.to(self.device)
        
        else:
            try:
                self.model = self.model.half().to(device)
                self.is_half = True
                self.dtype = torch.float16
            except:
                log.warning('显卡不支持半精度，改为全精度加载')
                self.model = self.model.to(self.device)
        self.model.eval()
        self.model_awake = True
        self.model_is_load = True
        
    def init_model(self):
        dict_s1 = torch.load(self.model_path, map_location="cpu")
        self.hz = 50
        self.config = dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]
        self.model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        self.model.load_state_dict(dict_s1["weight"])
        self.model_is_init = True
       
class Generator():
    def __init__(self,sovits_model,gpt_model,bert_model,ssl_model):
        self.sovits_model = sovits_model
        self.gpt_model = gpt_model
        self.bert_model = bert_model
        self.ssl_model = ssl_model
        self.reuse_objects = {}
        self.ref_wav_result = None
        self.phones1 = None
        self.bert1 = None 
        self.norm_text1 = None
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
        self.dict_language = {
                i18n("中文"): "all_zh",#全部按中文识别
                i18n("英文"): "en",#全部按英文识别#######不变
                i18n("日文"): "all_ja",#全部按日文识别
                i18n("中英混合"): "zh",#按中英混合识别####不变
                i18n("日英混合"): "ja",#按日英混合识别####不变
                i18n("多语种混合"): "auto",#多语种启动切分识别语种
            }
        
        if not self.sovits_model.model_is_load:
            self.sovits_model.load_model()
            # self.sovits_model.sleep_model()
        if not self.gpt_model.model_is_load:
            self.gpt_model.load_model()
            self.gpt_model.sleep_model()
        if not self.bert_model.model_is_load:
            self.bert_model.load_model()
            self.bert_model.sleep_model()
        if not self.ssl_model.model_is_load:
            self.ssl_model.load_model()
            self.ssl_model.sleep_model()
    
        if self.sovits_model.device == 'cuda' or self.gpt_model.device == 'cuda' or self.bert_model.device == 'cuda':
            self.final_device = 'cuda'
        else:
            self.final_device = 'cpu'
        if self.sovits_model.is_half or self.gpt_model.is_half or self.bert_model.is_half :
            self.final_half = True
            self.final_dtype = torch.float16
        else:
            self.final_device = False
            self.final_dtype = torch.float32        
        
        
    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free = False, stream=False):
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        t0 = ttime()
        prompt_language = self.dict_language[prompt_language]
        text_language = self.dict_language[text_language]
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in self.splits): prompt_text += "。" if prompt_language != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
        text = text.strip("\n")
        if (text[0] not in self.splits and len(self.get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
        
        print(i18n("实际输入的目标文本:"), text)
        zero_wav = np.zeros(
            int(self.sovits_model.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.sovits_model.is_half == True else np.float32
        )
        # 解析参考音频
        if self.ref_wav_result is None:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                    raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
                wav16k = torch.from_numpy(wav16k)
                zero_wav_torch = torch.from_numpy(zero_wav)
                if self.ssl_model.is_half == True:
                    wav16k = wav16k.half().to(self.ssl_model.device)
                    zero_wav_torch = zero_wav_torch.half().to(self.ssl_model.device)
                else:
                    wav16k = wav16k.to(self.ssl_model.device)
                    zero_wav_torch = zero_wav_torch.to(self.ssl_model.device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                self.ssl_model.awake_model()
                ssl_content = self.ssl_model.model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2).to(self.sovits_model.device).to(self.sovits_model.dtype)  # .float()
                self.ssl_model.sleep_model()
                codes = self.sovits_model.model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0].unsqueeze(0).to(self.gpt_model.device)
                del wav16k,zero_wav_torch,ssl_content,codes
                torch.cuda.empty_cache()
                self.ref_wav_result = prompt_semantic
        else:
            log.info('参考音频复用！')
            prompt_semantic = self.ref_wav_result
        t1 = ttime()

        if (how_to_cut == i18n("凑四句一切")):
            text = self.cut1(text)
        elif (how_to_cut == i18n("凑50字一切")):
            text = self.cut2(text)
        elif (how_to_cut == i18n("按中文句号。切")):
            text = self.cut3(text)
        elif (how_to_cut == i18n("按英文句号.切")):
            text = self.cut4(text)
        elif (how_to_cut == i18n("按标点符号切")):
            text = self.cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        if len(texts) == 1:
            text = self.cut3(texts[0])
            texts = text.split("\n")
        cut_depth = 0
        result_texts = []
        while True:
            end_loop = True
            result_texts = []
            for index,item in enumerate(texts):
                if len(item) > 30:
                    end_loop = False
                    if cut_depth == 0:
                        temp_text = self.cut1(item)
                        temp_list = temp_text.split("\n")
                        for inner_item in temp_list:
                            result_texts.append(inner_item)
                    elif cut_depth == 1:
                        temp_text = self.cut5(item)
                        temp_list = temp_text.split("\n")
                        for inner_item in temp_list:
                            result_texts.append(inner_item)
                    elif cut_depth == 2:
                        temp_text = self.cut5(temp_text)
                        temp_list = temp_text.split("\n")
                        for inner_item in temp_list:
                            result_texts.append(inner_item)
                    elif cut_depth == 3:
                        temp_text = self.cut2(item,40)
                        temp_list = temp_text.split("\n")
                        for inner_item in temp_list:
                            result_texts.append(inner_item)
                else:
                    result_texts.append(item)
            texts = result_texts
            if end_loop:
                break
            else:
                cut_depth += 1
        print(i18n("实际输入的目标文本(切句后):"), text)
        texts = text.split("\n")
        texts = self.merge_short_text_in_array(texts, 5)
        audio_opt = []
        if not ref_free:
            if self.phones1 is None or self.bert1 is None or self.norm_text1 is None:
                self.phones1,self.bert1,self.norm_text1=self.bert_model.get_phones_and_bert(prompt_text, prompt_language,self.bert_model.device,'cpu')
            else:
                log.info('参考音频文本解析复用！')
        self.gpt_model.awake_model()
        self.sovits_model.awake_model()
        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in self.splits): text += "。" if text_language != "en" else "."
            print(i18n("实际输入的目标文本(每句):"), text)
            phones2,bert2,norm_text2=self.bert_model.get_phones_and_bert(text, text_language,self.bert_model.device,self.final_device)
            print(i18n("前端处理后的文本(每句):"), norm_text2)
            if not ref_free:
                bert = torch.cat([self.bert1, bert2.to('cpu')], 1)
                all_phoneme_ids = torch.LongTensor(self.phones1+phones2).to(self.gpt_model.device).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(self.gpt_model.device).unsqueeze(0)
            bert = bert.unsqueeze(0).to(self.gpt_model.device).to(self.gpt_model.dtype)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.gpt_model.device).to(self.gpt_model.dtype)
            t2 = ttime()
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                
                pred_semantic, idx = self.gpt_model.model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt_semantic,
                    bert.to(self.gpt_model.device),
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.gpt_model.hz * self.gpt_model.max_sec,
                )
                del all_phoneme_ids,all_phoneme_len,bert
                torch.cuda.empty_cache()
            t3 = ttime()
            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(
                0
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = self.sovits_model.get_spepc(ref_wav_path)  # .to(device)
            if self.sovits_model.is_half == True:
                refer = refer.half().to(self.sovits_model.device)
            else:
                refer = refer.to(self.sovits_model.device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            phones2_tensor = torch.LongTensor(phones2).to(self.sovits_model.device).unsqueeze(0)
            audio = (
                self.sovits_model.model.decode(
                    pred_semantic.to(self.sovits_model.device), phones2_tensor, refer
                )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            del phones2_tensor,refer,pred_semantic
            torch.cuda.empty_cache()
            max_audio=np.abs(audio).max()#简单防止16bit爆音
            if max_audio>1:audio/=max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
            if stream:
                yield (np.concatenate([audio, zero_wav], 0) * 32768).astype(np.int16).tobytes()
        del prompt_semantic
        torch.cuda.empty_cache()
        self.gpt_model.sleep_model()
        self.sovits_model.sleep_model()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        if not stream:
            yield self.sovits_model.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
                np.int16
            )

    def split(self,todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    def cut1(self,inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
        else:
            opts = [inp]
        return "\n".join(opts)

    @staticmethod
    def cut2(inp,cut_num=30):
        inp = inp.strip("\n")
        count = 0
        opts = []
        tmp_str = ""
        for i in range(len(inp)):
            count += 1
            tmp_str += inp[i]
            if count >= cut_num:
                opts.append(tmp_str)
                count = 0
                tmp_str = ''
        if tmp_str != '':
            opts.append(tmp_str)
        return "\n".join(opts)

    @staticmethod
    def cut3(inp):
        inp = inp.strip("\n")
        return "\n".join(["%s" % item for item in inp.strip("。").split("。")])
    
    @staticmethod
    def cut4(inp):
        inp = inp.strip("\n")
        return "\n".join(["%s" % item for item in inp.strip(".").split(".")])

    @staticmethod
    # contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
    def cut5(inp):
        # if not re.search(r'[^\w\s]', inp[-1]):
        # inp += '。'
        inp = inp.strip("\n")
        punds = r'[,.;?!、，。？！;：…]'
        items = re.split(f'({punds})', inp)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items)%2 == 1:
            mergeitems.append(items[-1])
        opt = "\n".join(mergeitems)
        return opt

    @staticmethod
    def merge_short_text_in_array(texts, threshold):
        if (len(texts)) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if (len(text) > 0):
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result
    
    def get_first(self,text):
        pattern = "[" + "".join(re.escape(sep) for sep in self.splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text


# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


