import json
import io
import wave
from fastapi import FastAPI,Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import time
import torch
from GPT_SoVITS.generate_object import BertModel,CnhubertModel,SovitsModel,GptModel,Generator,cnhubert_base_path,bert_path,device,i18n

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

async def generate_tts_flow(generator,word,language,top_k,top_p,temperature,wave_head):
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
async def tts_flow_test(text:str):
    voice_id = '276'
    global generator
    if generator == None:
        generator = init_generator(voice_id)
    if generator.voice_id != voice_id:
        # 更换声音
        del generator
        torch.cuda.empty_cache()
        generator = init_generator(voice_id)
    return StreamingResponse(generate_tts_flow(generator,text,i18n("中英混合"),5,1,1,True), media_type="audio/x-wav")


def get_model_info(voice_id):
    
    sovits_path = '/root/GPT-SoVITS-Next/SoVITS_weights/276.pth'
    gpt_path = '/root/GPT-SoVITS-Next/GPT_weights/276.ckpt'
    prompt_info = {'ref_wav_path':'/root/GPT-SoVITS-Next/notype1 (1).wav',
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