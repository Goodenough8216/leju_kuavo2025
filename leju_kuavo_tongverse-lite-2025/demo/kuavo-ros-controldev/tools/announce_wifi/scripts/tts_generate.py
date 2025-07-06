#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from common.logger import KuavoLogger
class TTSGenerator:
    def __init__(self):
        # self._model_id = 'iic/speech_sambert-hifigan_tts_zhida_zh-cn_16k' # online
        self._model_id = '/opt/lejurobot/kuavo-wifi-announce/data/model/speech_sambert-hifigan_tts_zhida_zh-cn_16k' #offline
        self._sambert_hifigan_tts = None 
    def load_model(self):
        try:
            KuavoLogger.info("TTSGenerator Loading TTS Model: " + self._model_id)
            start_time = time.time()
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1,"
            self._sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech,  device='cpu', model=self._model_id)
            end_time = time.time()
            load_time = end_time - start_time
            KuavoLogger.info(f"TTSGenerator Loading model time cost: {load_time:.2f} seconds")
            return True
        except Exception as e:
            KuavoLogger.error(f"TTSGenerator Loading model error: {e}")
            self._sambert_hifigan_tts = None 
            return False
    def generate_text_to_speech(self, text:str, filepath:str)->bool:
        """
            text: input
            filepath: output file path
            return: True or False
        """
        if self._sambert_hifigan_tts is None:
            if not self.load_model():
                return  False
        try:
            KuavoLogger.info(f"TTSGenerator Generating TTS for: {text}")
            start_time = time.time()
            
            output = self._sambert_hifigan_tts(input=text)
            wav = output[OutputKeys.OUTPUT_WAV]  
            
            end_time = time.time()
            load_time = end_time - start_time
            KuavoLogger.info(f"TTSGenerator Generating TTS time cost: {load_time:.2f} seconds")

            # write to file.       
            with open(f'{filepath}', 'wb') as f:
                f.write(wav)
                return True
        except Exception as e:
            KuavoLogger.info(f"TTSGenerator generate Error: {e}")  
              
        return False