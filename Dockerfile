FROM python:3.10-slim
WORKDIR /app
VOLUME /app/data
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3","/app/make_submission.py"]

COPY . /app

RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip install gdown && \
    # Векторизаторы
    gdown --fuzzy --folder https://drive.google.com/drive/folders/1c6J7pLR6qtEfCh8Jp6JKZzkdywQLcZCR?usp=drive_link && \
    # Веса бертов
    gdown --fuzzy https://drive.google.com/file/d/1c9d03-pIwT5HJWfvEQ8PlxW5GtaEkuTB/view?usp=drive_link && \
    gdown --fuzzy https://drive.google.com/file/d/1vMe_znzoKJjUZ7gRRTQDpbch_5Nx98e6/view?usp=drive_link && \
    gdown --fuzzy https://drive.google.com/file/d/1GEI0lEi1gitio-aKdn0fdAni-sHMhZlB/view?usp=drive_link && \
    # делаем папку с бертами
    mkdir models && \
    mkdir models/BERT && \
    # Суем бертов в папку
    mv 3epoch_768_name_attr_bert_full.pth models/BERT/3epoch_768_name_attr_bert_full.pth && \
    mv 3epoch_1024_name_desc_bert_full.pth models/BERT/3epoch_1024_name_desc_bert_full.pth && \
    mv multi512_attr_bert_full_second_epoch.pth models/BERT/multi512_attr_bert_full_second_epoch.pth && \
    # катубсты 
    gdown --fuzzy --folder https://drive.google.com/drive/folders/1mktUxSWbg1YQHZXdSjQyBoSqwlD2pNdl?usp=drive_link && \
    # Перемещаем катбусты
    mv CATBOOST models/CATBOOST && \
    # Ставим либы
    pip install --no-cache-dir -r requirements.txt && \
    chmod +x /app/entrypoint.sh /app/make_submission.py && \
    # Качаем базовые модели с ХФ 
    huggingface-cli download cointegrated/rubert-tiny2 --local-dir='./models/basemodel/rubert' && \ 
    huggingface-cli download distilbert-base-multilingual-cased --local-dir='./models/basemodel/distilbert'