# Импортируем необходимые библиотеки и модули
from fastapi import FastAPI, Depends, HTTPException
from typing import List
from schema import UserGet, PostGet, FeedGet 
from sqlalchemy.orm import Session
from sqlalchemy.orm import relationship 
from sqlalchemy import create_engine
import os
from sqlalchemy import TIMESTAMP, Column, Integer, String, ForeignKey
from fastapi import FastAPI, Depends, HTTPException
from catboost import CatBoostClassifier
import pandas as pd
from datetime import datetime
from pydantic import BaseModel 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import hashlib
import logging

# Настройка подключения к базе данных
SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

# Создание двигателя и сессии для работы с базой данных
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для декларативных моделей базы данных
Base = declarative_base()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Определение моделей данных с использованием Pydantic
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class FeedGet(BaseModel): 
    user_id: int
    user: UserGet
    post_id: int
    post: PostGet 
    time: datetime 
    action: str

# Модель базы данных для таблицы "post"
class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

# Модель для ответа API
class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

# Функция для получения пути к модели в зависимости от окружения
def get_model_path(model_version: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f"/workdir/user_input/model_{model_version}"
    else:
        MODEL_PATH = f"C:\\my_porject\\model_{model_version}.cbm"
    return MODEL_PATH

# Функция для загрузки модели CatBoost
def load_models(model_name: str):
    model_path = get_model_path(model_name)
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    return from_file

# Функция для загрузки данных из SQL с использованием постраничной загрузки
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

# Функции для загрузки пользовательских и постовых фичей для тестовой и контрольной групп

def load_features_user_test() -> pd.DataFrame:
    query = "SELECT * FROM balashov_maxim_test_40_features_lesson_22"  
    return batch_load_sql(query)

def load_features_post_test() -> pd.DataFrame:
    query = "SELECT * FROM balashov_maxim_post_features_lesson_22_emb_47"  
    return batch_load_sql(query)

def load_features_user_control() -> pd.DataFrame:
    query = "SELECT * FROM balashov_maxim_test_13_features_lesson_22"  
    return batch_load_sql(query)

def load_features_post_control() -> pd.DataFrame:
    query = "SELECT * FROM balashov_maxim_post_features_lesson_22"  
    return batch_load_sql(query)

# Предзагрузка данных и моделей для минимизации задержек при запросах
features_post_control = load_features_post_control()
features_user_control = load_features_user_control()
model_control = load_models("control")

features_post_test = load_features_post_test()
features_user_test = load_features_user_test()
model_test = load_models("test")

# Функция для получения сессии базы данных
# Используется для управления подключением к базе данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Создание FastAPI приложения
app = FastAPI()

# Функция для определения экспериментальной группы пользователя
# Используется для разделения пользователей на "контроль" и "тест"
def get_exp_group(user_id: int) -> str:
    user_id_salt = str(user_id)+"my_salt"
    hashed_value = int(hashlib.md5((user_id_salt).encode()).hexdigest(), 16) % 100
    if hashed_value < 50:
        return "control"
    else:
        return "test"

# Основная логика рекомендаций для тестовой группы
# Здесь подгружаются фичи, делается предсказание модели и формируется список рекомендаций

def recommended_posts_with_test(id: int,  
                                time: datetime = datetime(year=2021, month=12, day=14, hour=14),  
                                limit: int = 5, 
                                db: Session = Depends(get_db)) -> List[PostGet]:
    
    query = """
        SELECT user_id, array_agg(DISTINCT post_id) AS post_ids
        FROM feed_action
        WHERE time <= :time_limit AND user_id = :user_id
        GROUP BY user_id
    """
    
    result = db.execute(text(query), {'user_id': id, 'time_limit': time}).fetchone()
    
    if not result or result.post_ids is None:
        post_ids = []  
    else:
        post_ids = result.post_ids  

    user_features = features_user_test[features_user_test['user_id'] == id]
    user_features = user_features.drop(["index", "user_id"], axis=1, errors='ignore')

    post_features = features_post_test[~features_post_test["post_id"].isin(post_ids)]
    post_id = post_features["post_id"]
    post_features = post_features.drop(["index", "post_id"], axis=1, errors='ignore')
    
    us_post = pd.concat((post_features, user_features), axis=1)
    
    features_cols = list(us_post.columns)
    for col in features_cols:
        if us_post[col].isna().any():
            value = us_post[col].dropna().iloc[0]
            us_post[col] = us_post[col].fillna(value)

    predictions = model_test.predict_proba(us_post)[:, 1]
    
    us_post["PROBA"] = predictions
    us_post["post_id"] = post_id
    
    top_posts = us_post.sort_values(by="PROBA", ascending=False).head(limit)
    recommended_posts_ids = list(top_posts["post_id"])
    
    recommended_posts = db.query(Post).filter(Post.id.in_(recommended_posts_ids)).all()

    return recommended_posts

# Основная логика рекомендаций для контрольной группы
# Аналогично тестовой группе, но с другими данными и моделью

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, 
                      time: datetime = datetime.now(), 
                      limit: int = 5, 
                      db: Session = Depends(get_db)):
    exp_group = get_exp_group(id)
    logger.info(f"User {id} assigned to group: {exp_group}")

    if exp_group == "control":
        recommendations = recommended_posts_with_control(id, time, limit, db)
    elif exp_group == "test":
        recommendations = recommended_posts_with_test(id, time, limit, db)
    else:
        logger.error(f"Unknown group for user {id}")
        raise HTTPException(status_code=400, detail="Unknown experimental group")

    return Response(exp_group=exp_group, recommendations=recommendations)
