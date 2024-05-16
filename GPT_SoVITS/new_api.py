import json
import io
import wave
import asyncio
from fastapi import FastAPI,Body,Request
from fastapi.responses import StreamingResponse
from requests.exceptions import ConnectionError
from pydantic import BaseModel
import uvicorn
import time
import torch
from generate_object import BertModel,CnhubertModel,SovitsModel,GptModel,Generator,cnhubert_base_path,bert_path,device,i18n

app = FastAPI()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

bert_device = device
ssl_device = device
sovits_device = device
gpt_device = device    

bert_model = BertModel(bert_path,bert_device)

ssl_model = CnhubertModel(cnhubert_base_path,ssl_device)

generator = None

class SpeakWordRequest(BaseModel):
    voice_id: str
    word: str
    language: str
    byte_stream: bool = True
    wave_head: bool = True
    how_to_cut: str = '不切'
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()

def generate_tts_flow(generator,word,language,top_k,top_p,temperature,wave_head):
    if generator.last_gen != word:
        generator.last_gen = word
        chunks = generator.get_tts_wav(ref_wav_path=generator.prompt_info['ref_wav_path'],
                            prompt_text=generator.prompt_info['prompt_text'],
                            prompt_language=generator.prompt_info['prompt_language'],
                            text=word,
                            text_language=language,
                            how_to_cut=i18n('不切'),
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            ref_free=False,
                            stream=True)
        if wave_head:
            yield wave_header_chunk()
            print("Head 发送")
        for chunk in chunks:
            yield chunk
    yield b''

@app.post('/tts_flow')
def tts_flow(input_params:SpeakWordRequest = Body(...)):
    global generator
    if generator == None:
        generator = init_generator(input_params.voice_id)
    if generator.voice_id != input_params.voice_id:
        # 更换声音
        del generator
        torch.cuda.empty_cache()
        generator = init_generator(input_params.voice_id)
    return StreamingResponse(generate_tts_flow(generator))


@app.get('/tts_flow_test')
def tts_flow_test(text:str,request:Request):
    voice_id = '276'
    global generator
    if generator == None:
        generator = init_generator(voice_id)
    if generator.voice_id != voice_id:
        # 更换声音
        del generator
        torch.cuda.empty_cache()
        generator = init_generator(voice_id)
    def generate_tts():
        try:
            for audio_chunk in generate_tts_flow(generator,text,i18n("中英混合"),5,1,1,True):
                if asyncio.run(request.is_disconnected()):
                    print("Client disconnected. Stopping TTS generation.")
                    break
                yield audio_chunk
        except ConnectionError:
            print("Connection error. Stopping TTS generation.")
    return StreamingResponse(generate_tts(),media_type="audio/x-wav")


def get_model_info(voice_id):
    
    sovits_path = r'C:\Users\Administrator\Downloads\GPT-SoVITS-Next\SoVITS_weights\276.pth'
    gpt_path = r'C:\Users\Administrator\Downloads\GPT-SoVITS-Next\GPT_weights\276.ckpt'
    prompt_info = {'ref_wav_path':r'C:\Users\Administrator\Downloads\GPT-SoVITS-Next\notype1.wav',
                   'prompt_text':'请帮我看一眼这封邮件，请给我来一份招牌菜。',
                   'prompt_language':i18n("中英混合"),
                   }
    return sovits_path,gpt_path,prompt_info

def init_generator(voice_id):
    sovits_path,gpt_path,prompt_info = get_model_info(voice_id)
    sovits_model = SovitsModel(sovits_path,device)
    gpt_model = GptModel(gpt_path,device)
    generator = Generator(sovits_model,gpt_model,bert_model,ssl_model)
    generator.voice_id = voice_id
    generator.prompt_info = prompt_info
    generator.last_gen = ''
    del sovits_model,gpt_model
    torch.cuda.empty_cache()
    return generator

if __name__ == '__main__':
    uvicorn.run(app='new_api:app', reload=True, host='0.0.0.0', port=8001)