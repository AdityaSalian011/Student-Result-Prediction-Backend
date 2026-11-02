from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer

from sqlalchemy.orm import Session
from sqlalchemy import text   ## CRON JOB
from database import get_db, Base, engine
from models import Teacher

from typing import Optional, Annotated
from pydantic import AfterValidator, EmailStr, BaseModel

from datetime import timedelta, datetime, timezone

import uuid, io, os
import pandas as pd
from jose import JWTError, jwt

from utils import get_student_feedback_reports, generate_html_report, get_synthetic_student_data, HTMLEmail, validate_dtypes

from weasyprint import HTML


SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES'))

FROM_EMAIL = os.getenv('FROM_EMAIL')
BREVO_API_KEY = os.getenv('BREVO_API_KEY')

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='login')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token
    - data: Dictionary containing teacher information (typically just teacher_id)
    - expires_delta: Token expiration time
    """
    to_encode = data.copy()

    if expires_delta:
        expires = datetime.now(timezone.utc) + expires_delta
    else:
        expires = datetime.now(timezone.utc) + timedelta(minutes=60)

    to_encode.update({
        'exp': expires
    })

    encoded_jwt = jwt.encode(to_encode, key=SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_teacher(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
):
    """
    Verify JWT token and return current teacher from database
    - This is used as a dependency for protected routes
    - Automatically validates token and fetches teacher data
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail='Could not validate credentials',
        headers={'WWW-Authenticate': 'Bearer'}
    )

    try:
        # Decode the jwt token
        payload = jwt.decode(token, key=SECRET_KEY, algorithms=[ALGORITHM])
        teacher_id = payload.get('sub')

        if not teacher_id:
            raise credentials_exception
        
        teacher_id = uuid.UUID(teacher_id)  ## convert str back to uuid for db query
    except JWTError:
        raise credentials_exception
    
    # Get the teacher from database
    teacher = db.query(Teacher).filter(
        Teacher.teacher_id == teacher_id
    ).first()

    if not teacher:
        raise credentials_exception
    
    return teacher

Base.metadata.create_all(bind=engine)

ALLOWED_DOMAINS = {
    'gmail.com', 'yahoo.com', 'outlook.com'
}

def popular_domains_only(value: str) -> str:
    domain = value.lower().split('@')[-1]  # extracting domain
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(
            f'Email domain @{domain} is not supported. '
            f'Please use one of: {", ".join(sorted(ALLOWED_DOMAINS))}'
        )
    return value.lower()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('FRONTEND_URL')],
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Authorization', 'Content-Type', 'Accept']
)

## HEALTH CHECK FOR CRON JOBS
@app.get('/health')
def health_check(db: Session = Depends(get_db)):
    try:
        # Run a lightweight query to keep Supabase awake
        db.execute(text('SELECT 1'))
        return {'status': 'ok', 'db': 'connected'}
    except Exception as e:
        return {'status': 'ok', 'db': 'error', 'detail': str(e)}

PopularEmailStr = Annotated[EmailStr, AfterValidator(popular_domains_only)]

class SignUpRequest(BaseModel):
    email: PopularEmailStr
    password: str 

@app.post('/signup')
async def signup(request: SignUpRequest, db: Session = Depends(get_db)):
    try:
        ## 1. Check if an teacher with same detail exists
        exists = db.query(Teacher).filter(
            Teacher.email==request.email
        ).first()
        if exists:
            raise HTTPException(
                status_code=400,
                detail=f'Teacher with email {request.email} already EXISTS..'
            )
        
        ## 2. If not create a new teacher data in Teacher table
        new_teacher = Teacher(
            email=request.email,
            password=request.password
        )

        ## 3. Add and commit new teacher detail to Teacher Table
        db.add(new_teacher)
        db.commit()
        db.refresh(new_teacher)

        # 4. 🔒 NEW: Create JWT token with only teacher ID
        # WHY: Token contains minimal data (just teacher_id) for security
        # Frontend uses this token to fetch teacher data when needed
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                'sub': str(new_teacher.teacher_id)
            }, # 'sub' is standard JWT claim for teacher identifier  
            ## sub should be an str not an int
            expires_delta=access_token_expires
        )

        ## 5. Return a sucess message to frontend (Debugging purpose only..)
        return{
            'message': 'New teacher added✅',
            'access_token': access_token,
            'token_type': 'bearer'
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f'Bad Gateway:\n{e}'
        )

class LoginRequest(BaseModel):
    email: PopularEmailStr
    password: str 

@app.post('/login')
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        ## 1. Check if an teacher with same detail exists
        exists = db.query(Teacher).filter(
            Teacher.email==request.email
        ).first()
        if not exists:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid email or password"
            )
        
        if request.password != exists.password:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid email or password"
            )
        
        ## 2. Create a new jwt token for the teacher
        access_token_expire = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={'sub': str(exists.teacher_id)},  
            ## sub is a standard key name to store id's in jwt token
            ## Stringify teacher id as int not accepted (causes error..)
            expires_delta=access_token_expire
        )

        ## 3. Return Message to frontend
        return {
            'message': 'Teacher Login Successful',
            'access_token': access_token,
            'token_type': 'bearer'
        }
        
    except HTTPException:
            raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f'Bad Gateway:\n{e}'
        )
    
class ForgotPasswordRequest(BaseModel):
    email: PopularEmailStr

# ============================================
# FORGOT PASSWORD ENDPOINT
# ============================================
@app.post('/forgot-password')
async def forgot_password(
    request: ForgotPasswordRequest,
    db: Session = Depends(get_db)
):
    teacher = db.query(Teacher).filter(
        Teacher.email == request.email
    ).first()

    ## Step1: Check if teacher exists
    if not teacher:
        raise HTTPException(
            status_code=404,
            detail=f'User with email {request.email} not found. Go to Signup page.'
        )
    
    ## Step2: Create shorter token
    reset_token = create_access_token(
        data={
            'sub': str(teacher.teacher_id),
            'purpose': 'password_reset'
        },
        expires_delta=timedelta(minutes=15)
    )
    # reset_link = f'http://127.0.0.1:5500/frontend/reset_password.html?token={reset_token}'
    reset_link = f'{os.getenv("FRONTEND_URL")}/reset_password.html?token={reset_token}'
    html_content = f"""
        <h2>Reset Your Password</h2>
        <p>Click the link below. It expires in 15 minutes.</p>
        <a href="{reset_link}" style="padding:10px 20px; background:#4CAF50; color:white; text-decoration:none; border-radius:5px;">
            Reset Password
        </a>
        <p>If you didn't request this, ignore this email.</p>
    """

    email_sender = HTMLEmail()
    email_sender.send_html_content(
        to_email=request.email,
        html_content=html_content,
        subject='Reset your EduPredict Password',
        from_email=FROM_EMAIL,
        brevo_api_key=BREVO_API_KEY
    )

    return {
        'message': f'A reset link has been sent at {request.email}'
    }

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

# ============================================
# RESET PASSWORD ENDPOINT
# ============================================
@app.post('/reset-password')
async def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=400,
        detail='Invalid or expired reset link'
    )

    try:
        payload = jwt.decode(request.token, key=SECRET_KEY, algorithms=[ALGORITHM])
        teacher_id = payload.get('sub')
        purpose = payload.get('purpose')

        if not teacher_id or purpose != 'password_reset':
            raise credentials_exception
        
    except JWTError:
        raise credentials_exception
    
    teacher = db.query(Teacher).filter(
        Teacher.teacher_id == uuid.UUID(teacher_id)
    ).first()

    if not teacher:
        raise credentials_exception
    
    teacher.password = request.new_password
    db.commit()

    return {
        'message': 'Password updated successfully.'
    }

@app.post('/upload-file')
async def receive_xlsx_file(
    file: UploadFile = File(...), 
    existing_teacher: Teacher = Depends(get_current_teacher)
    ):

    try:
        if not file.filename.endswith('xlsx'):
            raise HTTPException(
                status_code=400,
                detail=f'Only Excel files are accepted'
            )
    
        content = await file.read()    # reads the raw bytes FastAPI received
        buffer = io.BytesIO(content)   # wraps bytes into a fake in-memory file
        df = pd.read_excel(buffer, na_values=['na', 'Na', 'null', 'NaN', 'None', 'NA', 'N/A'])     # pandas reads it as if it were a real .xlsx on disk

        invalid_values, missing_values = validate_dtypes(df)

        if invalid_values:

            joined = "\n".join(invalid_values)
            error_msg = f"{joined}\nDownload the sample file to see the correct format."

            raise HTTPException(
                status_code=400,
                detail=error_msg
            )   
        
        headers = {'Content-Disposition': 'attachment; filename="student_report.pdf"'}
        # ↑ tells browser "treat this as a download, not a page"

        if missing_values:
            headers['X-Missing-Values'] = missing_values
            headers['Access-Control-Expose-Headers'] = 'X-Missing-Values'

        reports = get_student_feedback_reports(df)

        html_report = generate_html_report(student_report=reports)

        pdf_buffer = io.BytesIO()
        HTML(string=html_report).write_pdf(pdf_buffer)   # writes PDF bytes into memory
        pdf_buffer.seek(0)   #  resets cursor to start so it can be read

        return StreamingResponse(
            pdf_buffer,    # streams bytes directly to browser 
            media_type='application/pdf',   # tells browser "this is a PDF"
            headers=headers
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f'Bad Gateway\n{e}'
        )

@app.get('/get-sample')
async def get_student_sample():
    try:
        synthetic_df = get_synthetic_student_data()

        excel_buffer = io.BytesIO()
        synthetic_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        return StreamingResponse(
            excel_buffer,    # streams bytes directly to browser 
            media_type='application/xlsx',   # tells browser "this is a Excel File"
            headers={'Content-Disposition': 'attachment; filename="student_sample.xlsx"'}
            # ↑ tells browser "treat this as a download, not a page"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f'Bad Gateway\n{e}'
        )
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)