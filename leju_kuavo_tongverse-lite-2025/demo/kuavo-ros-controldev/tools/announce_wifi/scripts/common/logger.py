import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    logger = logging.getLogger('kuavo-wifi-announce')
    logger.setLevel(logging.DEBUG)
     
    log_dir = '/opt/lejurobot/kuavo-wifi-announce/log/'
    os.system(f'mkdir -p {log_dir}')
    
    log_file = f'{log_dir}/kuavo-wifi-announce.log'
    fh = RotatingFileHandler(log_file, maxBytes=2*1024*1024, backupCount=5)  # 每个日志文件最大 2 MB，保留 5 个备份文件
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

KuavoLogger = setup_logger()