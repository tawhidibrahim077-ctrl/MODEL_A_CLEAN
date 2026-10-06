"""
Model A: prottek user er alada account, alada PDF, alada vector database.
PDF er original copy Cloudinary te (private) save hoy.

Chalate:  uvicorn model_a:app --reload
"""
import io
import os
import re
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import cloudinary
import cloudinary.uploader
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ========== CONFIG ==========
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
TOKEN_HOURS = int(os.getenv("TOKEN_HOURS", str(24 * 7)))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUD_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUD_SECRET = os.getenv("CLOUDINARY_API_SECRET")

if not GOOGLE_API_KEY:
    raise RuntimeError(".env e GOOGLE_API_KEY nai.")
if len(JWT_SECRET) < 32:
    raise RuntimeError(
        ".env e JWT_SECRET nai ba khub chhoto (kom pokkhe 32 character). Toiri korte:\n"
        '  python -c "import secrets; print(secrets.token_hex(32))"'
    )
if not (CLOUD_NAME and CLOUD_KEY and CLOUD_SECRET):
    raise RuntimeError(".env e CLOUDINARY_CLOUD_NAME / CLOUDINARY_API_KEY / CLOUDINARY_API_SECRET lagbe.")

cloudinary.config(cloud_name=CLOUD_NAME, api_key=CLOUD_KEY, api_secret=CLOUD_SECRET, secure=True)

BASE_DIR = Path(__file__).resolve().parent
STORAGE = BASE_DIR / "storage"
CHROMA_ROOT = STORAGE / "chroma"
DB_FILE = STORAGE / "users.db"
CHROMA_ROOT.mkdir(parents=True, exist_ok=True)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")

app = FastAPI(title="RAG API (Model A)")

# Token Authorization header e jay (cookie na), tai allow_credentials lage na.
# Production e "*" er bodole nijer website er address din.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== DATABASE (SQLite) ==========
@contextmanager
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


init_db()

# ========== AUTH ==========
ph = PasswordHasher()
DUMMY_HASH = ph.hash("dummy-password")
bearer = HTTPBearer(auto_error=False)


def make_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(401, "Age login korun")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(401, "Session shesh, abar login korun")
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(401, "Account paoa jay ni, abar login korun")
    return dict(row)


class RegisterIn(BaseModel):
    email: str
    username: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class AskIn(BaseModel):
    question: str


def public_user(u: dict) -> dict:
    return {"id": u["id"], "email": u["email"], "username": u["username"]}


@app.get("/")
def root():
    return {"status": "ok", "app": "RAG API (Model A)"}


@app.post("/auth/register")
def register(data: RegisterIn):
    email = data.email.strip().lower()
    username = data.username.strip()
    password = data.password

    if not EMAIL_RE.match(email) or len(email) > 254:
        raise HTTPException(400, "Email thik na")
    if not USERNAME_RE.match(username):
        raise HTTPException(400, "Username 3-30 character, shudhu english letter, number, _ . - cholbe")
    if len(password) < 8:
        raise HTTPException(400, "Password kom pokkhe 8 character hote hobe")
    if len(password) > 128:
        raise HTTPException(400, "Password onek boro")

    try:
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, username, password_hash) VALUES (?, ?, ?)",
                (email, username, ph.hash(password)),
            )
            user_id = cur.lastrowid
    except sqlite3.IntegrityError as e:
        if "email" in str(e):
            raise HTTPException(409, "Ei email diye already account ache")
        raise HTTPException(409, "Ei username already newa hoye gese")

    user = {"id": user_id, "email": email, "username": username}
    return {"token": make_token(user_id), "user": user}


@app.post("/auth/login")
def login(data: LoginIn):
    email = data.email.strip().lower()
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    bad = HTTPException(401, "Email ba password vul")
    if row is None:
        try:
            ph.verify(DUMMY_HASH, data.password)
        except Exception:
            pass
        raise bad
    try:
        ph.verify(row["password_hash"], data.password)
    except (VerificationError, InvalidHashError):
        raise bad

    user = dict(row)
    return {"token": make_token(user["id"]), "user": public_user(user)}


@app.get("/me")
def me(user: dict = Depends(get_current_user)):
    return public_user(user)


# ========== RAG HELPERS (prottek user er alada) ==========
def safe_name(name: str) -> str:
    return Path(name or "").name


def cloud_public_id(user: dict, name: str) -> str:
    """Cloudinary te file er naam. Special character bad dei, kintu same function
    upload ar delete duto-te use hoy, tai shobshomoy same id pabe."""
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return f"model_a/user_{user['id']}/{clean}"


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY)


def get_vectorstore(user: dict):
    path = CHROMA_ROOT / f"user_{user['id']}"
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=f"docs_u{user['id']}",
        persist_directory=str(path),
        embedding_function=get_embeddings(),
    )


def list_sources(vs) -> dict:
    try:
        data = vs._collection.get(include=["metadatas"])
    except Exception:
        return {}
    counts = {}
    for m in data.get("metadatas") or []:
        src = (m or {}).get("source")
        if src:
            counts[src] = counts.get(src, 0) + 1
    return counts


def remove_vectors(user: dict, name: str) -> bool:
    try:
        vs = get_vectorstore(user)
        if name in list_sources(vs):
            vs._collection.delete(where={"source": name})
            return True
    except Exception as e:
        print("REMOVE VECTORS ERROR:", repr(e))
    return False


def to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def is_quota_error(e: Exception) -> bool:
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def process_pdf(pdf_path, user: dict, display_name: Optional[str] = None) -> int:
    pdf_path = Path(pdf_path)  # str dilew cholbe
    source_name = display_name or pdf_path.name  # asol PDF er naam, temp file er na

    docs = PyPDFLoader(str(pdf_path)).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError("PDF theke text pawa jay ni (scanned/image PDF hote pare)")
    for c in chunks:
        c.metadata["source"] = source_name

    vs = get_vectorstore(user)
    try:
        vs._collection.delete(where={"source": source_name})
    except Exception:
        pass

    BATCH = 20
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        for _ in range(5):
            try:
                vs.add_documents(batch)
                break
            except Exception as e:
                if is_quota_error(e):
                    time.sleep(45)
                else:
                    raise
        else:
            raise RuntimeError("Google quota bar bar shesh hocche, kichukkhon pore abar try korun")
        if i + BATCH < len(chunks):
            time.sleep(13)
    return len(chunks)


def answer_question(vs, question: str):
    docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "Tumi ekta helpful assistant. Shudhu niche deya context theke uttor dao. "
        "Context e uttor na thakle bolo 'PDF e eta pai nai'.\n\n"
        f"Context:\n{context}\n\nProshno: {question}\n\nUttor:"
    )
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)
    response = llm.invoke(prompt)

    seen, sources = set(), []
    for d in docs:
        key = (d.metadata.get("source", ""), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        page = key[1]
        sources.append({"file": key[0], "page": (page + 1) if isinstance(page, int) else None})
    return to_text(response.content), sources


# ========== PDF ROUTES (shob login lage) ==========
@app.post("/upload")
def upload_pdf(file: List[UploadFile] = File(...), user: dict = Depends(get_current_user)):
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    results = []

    for f in file:
        name = safe_name(f.filename)
        if not name.lower().endswith(".pdf"):
            results.append({"file": name, "status": "error", "detail": "Shudhu PDF file cholbe"})
            continue
        data = f.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            results.append({"file": name, "status": "error", "detail": f"File {MAX_UPLOAD_MB}MB er beshi"})
            continue
        if not data.startswith(b"%PDF-"):
            results.append({"file": name, "status": "error", "detail": "Eta asol PDF na"})
            continue

        tmp_path = None
        try:
            # 1. memory er data theke temp file (Cloudinary theke abar download lage na)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            # file ekhon close, tai Windows e o PyPDFLoader porte parbe

            # 2. process: chunk + embedding
            n = process_pdf(tmp_path, user, display_name=name)

            # 3. process shofol hole tobei Cloudinary te private copy rakhi
            cloudinary.uploader.upload(
                io.BytesIO(data),
                resource_type="raw",
                type="authenticated",  # public link na, keu URL jante parle o dekhte parbe na
                public_id=cloud_public_id(user, name),
                overwrite=True,
                invalidate=True,
            )
            results.append({"file": name, "status": "ok", "chunks": n})
        except (ValueError, RuntimeError) as e:
            remove_vectors(user, name)
            results.append({"file": name, "status": "error", "detail": str(e)})
        except Exception as e:
            remove_vectors(user, name)
            print("UPLOAD ERROR:", name, repr(e))
            detail = "Google quota shesh, pore try korun" if is_quota_error(e) else "Upload/process korte problem hoise"
            results.append({"file": name, "status": "error", "detail": detail})
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]
    if not ok and failed:
        raise HTTPException(400, failed[0]["detail"])
    return {"status": "ok", "ok": len(ok), "failed": len(failed), "results": results}


@app.get("/list_pdfs")
def list_pdfs(user: dict = Depends(get_current_user)):
    return {"files": sorted(list_sources(get_vectorstore(user)).keys())}


@app.delete("/delete_pdf/{filename}")
def delete_pdf(filename: str, user: dict = Depends(get_current_user)):
    filename = safe_name(filename)
    found = remove_vectors(user, filename)

    try:
        r = cloudinary.uploader.destroy(
            cloud_public_id(user, filename),
            resource_type="raw",
            type="authenticated",
            invalidate=True,
        )
        if r.get("result") == "ok":
            found = True
    except Exception as e:
        print("CLOUDINARY DELETE ERROR:", repr(e))

    return {"status": "deleted" if found else "not found"}


@app.post("/ask")
def ask(data: AskIn, user: dict = Depends(get_current_user)):
    question = data.question.strip()
    if not question:
        raise HTTPException(400, "Proshno likhun")
    if len(question) > 2000:
        raise HTTPException(400, "Proshno onek boro (2000 character er moddhe)")
    try:
        vs = get_vectorstore(user)
        if not list_sources(vs):
            return {"answer": "Age PDF upload koro", "sources": []}
        answer, sources = answer_question(vs, question)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print("ASK ERROR:", repr(e))
        if is_quota_error(e):
            raise HTTPException(429, "Google quota shesh, kichukkhon pore abar try korun")
        raise HTTPException(500, "Uttor toiri korte problem hoise")