from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
import uuid
from database import Base

class Teacher(Base):
    __tablename__ = 'teachers'

    teacher_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True)
    password = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f'<User(id={self.teacher_id}, email={self.email})>'