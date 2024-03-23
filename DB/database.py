from sqlalchemy import create_engine, Column, Integer, String, Text, PrimaryKeyConstraint ,Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

Base = declarative_base()

class User(Base):
    __tablename__ = 'chat_sessions'

    speaker_id = Column(String, nullable=False)
    session_id = Column(Integer, nullable=False)
    time_stamp = Column(Integer, nullable=False)
    use_whisper = Column(Boolean, default=False)
    llm_model_name = Column(String)
    question = Column(Text)
    answer = Column(Text)
    

    __table_args__ = (
        PrimaryKeyConstraint('speaker_id', 'session_id'),
        {},
    )

class Database:
    def __init__(self,db_url):
        self.engine = create_engine(db_url, echo=False , pool_timeout=1000)
        self.session = scoped_session(sessionmaker(bind=self.engine))
        Base.metadata.create_all(self.engine)
    
    def add_session(self,speaker_id,time_stamp,use_whisper,llm_model_name,question ='',answer = ''):
        # user = User(speaker_id = speaker_id,session_id=session_id,context=context)
        latest_session_id = self.session.query(User).filter_by(speaker_id=speaker_id).count()
        session_id = latest_session_id + 1
        user = self.session.query(User).filter_by(speaker_id=speaker_id, session_id=session_id).first()
        
        if user: # if user exists update the context #基本的にあり得ないが保険
            user.time_stamp = time_stamp
            user.use_whisper = use_whisper
            user.llm_model_name = llm_model_name
            user.question = question
            user.answer = answer

        else:
            user = User(speaker_id = speaker_id,session_id=session_id,time_stamp=time_stamp,use_whisper=use_whisper,llm_model_name=llm_model_name,question =question,answer = answer)
            self.session.add(user)


        self.session.commit()

    def get_session(self,speaker_id,session_id):
        user = self.session.query(User).filter_by(speaker_id = speaker_id,session_id=session_id).first()
        return user
    
    def get_memory(self,speaker_id,memory_num):
        latest_session_id = self.session.query(User).filter_by(speaker_id=speaker_id).count()
        question = []
        answer = []

        for id in range(latest_session_id,latest_session_id-memory_num,-1):
            user = self.session.query(User).filter_by(speaker_id = speaker_id,session_id=id).first()
            if user:
                question.append(user.question)
                answer.append(user.answer)


        return question,answer
    
    # def update_session(self,speaker_id,session_id,time_stamp,context):
    #     user = self.session.query(User).filter_by(speaker_id = speaker_id,session_id=session_id).first()
    #     if user:
    #         user.time_stamp = time_stamp
    #         user.context = context
    #         self.session.commit()


    def fetch_all_sessions(self):
        sessions = self.session.query(User).all()

        return [
        {   
            "speaker_id": session.speaker_id, 
            "session_id": session.session_id, 
            "time_stamp": session.time_stamp,
            "use_whisper": session.use_whisper,
            "llm_model_name": session.llm_model_name,
            "question": session.question,
            "answer": session.answer,
        }

        for session in sessions
        ]
