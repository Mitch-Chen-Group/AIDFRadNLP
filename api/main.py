from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from rad_nlp_pipeline import run_pipeline_on_reports
from api import database, auth

database.init_db()

app = FastAPI(title="AIDFRadNLP API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ---------- Schemas ----------
class ProcessRequest(BaseModel):
    reports: list[str]
    deid: bool = False

class UserCreate(BaseModel):
    username: str
    password: str

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = auth.decode_token(token)
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user = db.query(database.User).filter(database.User.username == username).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------- Auth Routes ----------
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    hashed_pw = auth.hash_password(user.password)
    db_user = database.User(username=user.username, hashed_password=hashed_pw)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"username": db_user.username}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(database.User).filter(database.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = auth.create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# ---------- Pipeline Route ----------
@app.post("/process")
def process_reports(req: ProcessRequest, user = Depends(get_current_user)):
    results = run_pipeline_on_reports(req.reports, req.deid)
    return {"results": results}
