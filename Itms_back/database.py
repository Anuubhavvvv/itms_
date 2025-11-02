# database.py
# ---------------------------------------------------------------------
# SQLAlchemy setup for PostgreSQL
# ---------------------------------------------------------------------

from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import DB_URL

Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    vehicle_conf = Column(Float)
    plate_conf = Column(Float)
    ocr_text = Column(String)
    vehicle_path = Column(String)
    plate_path = Column(String)

# Initialize DB engine and session
if not DB_URL:
    print("[WARNING] DB_URL in config.py is empty — fill it in before running.")
engine = create_engine(DB_URL or "sqlite:///fallback.db", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
