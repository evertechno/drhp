"""
streamlit_app.py

Streamlit app that:
- Allows sign-in via Supabase email/password
- Uploads DRHP docs to Supabase Storage
- Extracts text from uploaded files (PDF/DOCX/TXT)
- Persists metadata to `clustersync_documents` table
- Triggers AI analysis either by calling:
    1) your existing edge function (preferred) via EDGE_FUNCTION_URL; OR
    2) direct Gemini/PaLM API if GEMINI_API_KEY is provided (basic fallback)
- Shows analysis results.

Requirements:
- Set environment variables (see README section below).
- Run with: streamlit run streamlit_app.py
"""

import os
import time
import json
import io
import math
import pathlib
from typing import Optional, Dict, Any

import streamlit as st
from supabase import create_client, Client
import requests

# Document processing libraries
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# ---------- Configuration & env ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")  # optional for some server-side actions
SUPABASE_STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "documents")
EDGE_FUNCTION_URL = os.environ.get("EDGE_FUNCTION_URL", "")  # If set, app will call this to run analysis (preferred)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # optional fallback direct call
MAX_EXTRACT_CHARS = int(os.environ.get("MAX_EXTRACT_CHARS", "1000000"))  # cap stored extracted content

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.warning("SUPABASE_URL and SUPABASE_ANON_KEY must be set as environment variables.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# ---------- Utility functions ----------
def supabase_signin(email: str, password: str) -> Dict[str, Any]:
    """
    Sign in with email+password. Returns dict with 'user' and 'session' or throws.
    """
    resp = supabase.auth.sign_in_with_password({"email": email, "password": password})
    # supabase-py returns dict-like response; adapt depending on library version
    if isinstance(resp, dict) and (resp.get("error") or resp.get("user") is None):
        # older client style
        raise Exception(resp.get("error", "Sign in failed"))
    return resp


def supabase_signup(email: str, password: str) -> Dict[str, Any]:
    resp = supabase.auth.sign_up({"email": email, "password": password})
    if isinstance(resp, dict) and (resp.get("error") or resp.get("user") is None):
        raise Exception(resp.get("error", "Sign up failed"))
    return resp


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Return supabase auth user if signed in in this session
    """
    try:
        # new supabase-py: auth.get_user()
        result = supabase.auth.get_user()
        user = None
        if isinstance(result, dict):
            # client may return dict with 'data' key
            user = result.get("data", {}).get("user")
        else:
            # some versions return object with data
            user = getattr(result, "user", None) or getattr(result, "data", {}).get("user")
        return user
    except Exception:
        # fallback to user stored in session_state
        return st.session_state.get("user")


def upload_file_to_storage(file_bytes: bytes, filename: str, user_id: str) -> Dict[str, Any]:
    """
    Uploads file to Supabase Storage bucket. Returns public url + metadata.
    """
    key = f"{user_id}/{int(time.time())}_{filename}"
    # Supabase Storage uses client.storage.from(bucket).upload()
    storage = supabase.storage.from_(SUPABASE_STORAGE_BUCKET)
    try:
        res = storage.upload(key, io.BytesIO(file_bytes))
    except Exception as e:
        # some versions require different call signatures
        res = storage.upload(key, file_bytes)
    # create public URL
    public_url = storage.get_public_url(key).get("publicURL")
    return {"path": key, "public_url": public_url}


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes using PyMuPDF if available, otherwise returns empty string.
    """
    if fitz is None:
        st.warning("PyMuPDF (fitz) not installed — PDF extraction will be limited. Install 'pymupdf'.")
        return ""
    text_chunks = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text_chunks.append(page.get_text("text"))
        return "\n\n".join(text_chunks)
    except Exception as e:
        st.error(f"Failed to extract PDF text: {e}")
        return ""


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    if docx is None:
        st.warning("python-docx not installed — DOCX extraction will be limited. Install 'python-docx'.")
        return ""
    try:
        bio = io.BytesIO(docx_bytes)
        document = docx.Document(bio)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        st.error(f"Failed to extract DOCX text: {e}")
        return ""


def persist_document_metadata(user_id: str, file_name: str, file_size: int, storage_path: str,
                              extracted_content: str, processing_status: str = "uploaded") -> Dict[str, Any]:
    """
    Inserts a row into clustersync_documents and returns the new record.
    Expects a table `clustersync_documents` to exist with columns matching the Deno function.
    """
    # trim extracted content to MAX_EXTRACT_CHARS to avoid huge payloads
    ec = extracted_content or ""
    if len(ec) > MAX_EXTRACT_CHARS:
        ec = ec[:MAX_EXTRACT_CHARS]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "user_id": user_id,
        "file_name": file_name,
        "file_size": file_size,
        "storage_path": storage_path,
        "processing_status": processing_status,
        "extracted_content": ec,
        "uploaded_at": now
    }
    # Use insert with returning single
    resp = supabase.table("clustersync_documents").insert(payload).select("*").execute()
    # supabase-py returns a Result object with data
    if hasattr(resp, "data"):
        data = resp.data
    else:
        data = resp
    # Data may be a list
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return data


def create_analysis_record(user_id: str, document_id: int, entity_type: str = "IPO_COMPLIANCE", query: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "user_id": user_id,
        "document_id": document_id,
        "selected_entity_type": entity_type,
        "query_used": query or "",
        "analysis_status": "processing",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    resp = supabase.table("clustersync_analysis_results").insert(payload).select("*").execute()
    if hasattr(resp, "data"):
        data = resp.data
    else:
        data = resp
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return data


def update_analysis_record(analysis_id: int, updates: dict) -> Dict[str, Any]:
    resp = supabase.table("clustersync_analysis_results").update(updates).eq("id", analysis_id).select("*").execute()
    if hasattr(resp, "data"):
        data = resp.data
    else:
        data = resp
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return data


def call_edge_function(document_id: int, entity_type: str, query: Optional[str], token: Optional[str]) -> Dict[str, Any]:
    """
    Call your existing edge function (Deno) that expects {documentId, entityType, query} JSON
    and Authorization header with Supabase session token. Returns parsed JSON.
    """
    if not EDGE_FUNCTION_URL:
        raise Exception("EDGE_FUNCTION_URL is not set. Configure your edge function URL in env.")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = token
    payload = {"documentId": document_id, "entityType": entity_type, "query": query}
    resp = requests.post(EDGE_FUNCTION_URL, json=payload, headers=headers, timeout=600)
    if not resp.ok:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise Exception(f"Edge function error {resp.status_code}: {err}")
    return resp.json()


def call_gemini_direct(document_text: str, file_name: str, entity_type: str, document_metadata: dict, comprehensive_regulatory_data: list) -> dict:
    """
    Minimal direct call to Gemini/PaLM-style API for fallback. Uses GEMINI_API_KEY.
    This function mirrors the Deno call pattern but simplified.
    NOTE: You must ensure allowed network access and that GEMINI_API_KEY is valid.
    """
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not set.")
    # Build a simple prompt (we keep it shorter than the Deno version for brevity)
    prompt = f"""As a senior regulatory compliance expert specializing in {entity_type}, analyze the document below and return VALID JSON only with keys:
compliance_score, risk_assessment, identified_gaps, recommendations, regulatory_references, document_analysis, executive_summary, next_steps.

Document metadata: {json.dumps(document_metadata)}
Document content (first 20000 chars):
{document_text[:20000]}
"""
    # Simple HTTP call to Google Generative Language / Gemini endpoint
    # The exact endpoint and headers may differ depending on the API you're using.
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 16384,
            "topK": 40,
            "topP": 0.9,
            "responseMimeType": "application/json"
        }
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    resp = requests.post(endpoint, json=body, headers=headers, timeout=900)
    if not resp.ok:
        raise Exception(f"Gemini API call failed: {resp.status_code} {resp.text}")
    data = resp.json()
    # attempt to extract candidate text and parse JSON
    try:
        analysis_text = data.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text")
    except Exception:
        analysis_text = None
    if not analysis_text:
        raise Exception("Gemini returned no analysis text.")
    # try to parse the JSON block inside analysis_text
    # coarse cleanup
    try:
        jstart = analysis_text.index("{")
        jend = analysis_text.rindex("}")
        json_text = analysis_text[jstart:jend+1]
        parsed = json.loads(json_text)
        return parsed
    except Exception as e:
        # return raw text as fallback
        return {"raw_text": analysis_text, "error_parse": str(e)}


# ---------- Streamlit UI ----------
st.set_page_config(page_title="DRHP Analyzer (Supabase + AI)", page_icon="📄", layout="wide")

st.title("📄 DRHP Analyzer — Supabase + AI")

# Sidebar: Authentication
with st.sidebar:
    st.header("Account")
    if "user" not in st.session_state:
        st.session_state.user = None

    if st.session_state.get("user"):
        user = st.session_state.user
        st.markdown(f"**Signed in as:** {user.get('email')}")
        if st.button("Sign out"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.user = None
            st.experimental_rerun()
    else:
        auth_tab = st.radio("Login / Sign up", ("Login", "Sign up"))
        email = st.text_input("Email", key="email_input")
        password = st.text_input("Password", type="password", key="password_input")
        if auth_tab == "Login":
            if st.button("Sign in"):
                try:
                    resp = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    # Adapt to different return shapes
                    if isinstance(resp, dict) and resp.get("error"):
                        st.error(resp.get("error"))
                    else:
                        # store user in session
                        user = None
                        if isinstance(resp, dict):
                            user = resp.get("user") or resp.get("data", {}).get("user")
                        else:
                            user = getattr(resp, "user", None) or getattr(resp, "data", {}).get("user")
                        st.session_state.user = user
                        st.success("Signed in")
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Sign-in failed: {e}")
        else:  # Sign up
            if st.button("Create account"):
                try:
                    resp = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Sign-up OK. Check your email for confirmation if required.")
                except Exception as e:
                    st.error(f"Sign-up failed: {e}")

# After login, show upload & documents
user = st.session_state.get("user") or get_current_user()
if user:
    st.sidebar.markdown("---")
    st.sidebar.write("Supabase user ID:")
    st.sidebar.code(user.get("id"))

    st.header("Upload DRHP / Document")
    uploaded_file = st.file_uploader("Upload DRHP (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=False)

    entity_type = st.selectbox("Select entity type for analysis", ["IPO_COMPLIANCE", "AIF", "PMS", "MUTUAL_FUND", "INVESTMENT_ADVISER", "STOCK_BROKER", "MERCHANT_BANKER"], index=0)
    custom_query = st.text_area("Optional: add a custom query to guide analysis", height=80)

    if uploaded_file:
        st.info(f"Received file: {uploaded_file.name} ({uploaded_file.size} bytes)")
        if st.button("Upload & Process"):
            # Read bytes
            file_bytes = uploaded_file.read()
            # upload to storage
            with st.spinner("Uploading file to Supabase storage..."):
                try:
                    upload_meta = upload_file_to_storage(file_bytes, uploaded_file.name, user.get("id"))
                    st.success("Uploaded to storage.")
                except Exception as e:
                    st.error(f"Storage upload failed: {e}")
                    st.stop()

            # extract text depending on type
            ext = pathlib.Path(uploaded_file.name).suffix.lower()
            extracted_text = ""
            with st.spinner("Extracting text..."):
                if ext == ".pdf":
                    extracted_text = extract_text_from_pdf_bytes(file_bytes)
                elif ext == ".docx":
                    extracted_text = extract_text_from_docx_bytes(file_bytes)
                elif ext == ".txt":
                    try:
                        extracted_text = file_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        extracted_text = file_bytes.decode("latin-1", errors="replace")
                else:
                    st.warning("Unknown extension — attempting to decode as text.")
                    try:
                        extracted_text = file_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        extracted_text = ""

            if not extracted_text:
                st.warning("No extracted text found — you can still store and run analysis, but results may be limited.")

            # persist metadata & extracted content
            with st.spinner("Saving document metadata to database..."):
                try:
                    doc_record = persist_document_metadata(
                        user_id=user.get("id"),
                        file_name=uploaded_file.name,
                        file_size=uploaded_file.size,
                        storage_path=upload_meta["path"],
                        extracted_content=extracted_text,
                        processing_status="uploaded"
                    )
                    st.success("Document metadata saved.")
                except Exception as e:
                    st.error(f"Failed to save metadata: {e}")
                    st.stop()

            st.session_state["latest_document"] = doc_record
            st.experimental_rerun()

    # list recent documents for user
    st.header("Your documents")
    try:
        res = supabase.table("clustersync_documents").select("*").eq("user_id", user.get("id")).order("uploaded_at", desc=True).limit(20).execute()
        docs = res.data if hasattr(res, "data") else res
    except Exception:
        docs = []
    if not docs:
        st.info("No documents uploaded yet.")
    else:
        for d in docs:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{d.get('file_name')}**  \nUploaded: {d.get('uploaded_at')}  \nStatus: {d.get('processing_status')}")
                if d.get("extracted_content"):
                    preview = d.get("extracted_content")[:1000].replace("\n", " ")
                    st.write(preview + ("..." if len(d.get("extracted_content")) > 1000 else ""))
            with col2:
                if st.button("Run analysis", key=f"analyze_{d.get('id')}"):
                    # create analysis record
                    try:
                        analysis_record = create_analysis_record(user.get("id"), d.get("id"), entity_type=entity_type, query=custom_query)
                    except Exception as e:
                        st.error(f"Failed to create analysis record: {e}")
                        st.stop()

                    st.session_state["current_analysis_id"] = analysis_record.get("id")
                    st.session_state["current_document"] = d
                    # Kick off processing inline (synchronous)
                    st.experimental_rerun()
            with col3:
                if d.get("storage_path"):
                    st.markdown(f"[Open file in storage] (via Supabase Storage path)")
    st.markdown("---")

    # If we kicked off an analysis, show processing panel and run it
    if st.session_state.get("current_analysis_id") and st.session_state.get("current_document"):
        analysis_id = st.session_state["current_analysis_id"]
        doc = st.session_state["current_document"]
        st.header("Analysis in progress")
        st.write(f"Document: {doc.get('file_name')}  — Analysis ID: {analysis_id}")
        # If analysis_status is processing in DB, we run analysis now (synchronous)
        # Fetch freshest document row
        try:
            fresh = supabase.table("clustersync_documents").select("*").eq("id", doc.get("id")).single().execute()
            fresh_doc = fresh.data if hasattr(fresh, "data") else fresh
        except Exception:
            fresh_doc = doc
        st.write("Processing... (this runs synchronously in the Streamlit app — may take minutes)")

        # Execute analysis path
        try:
            token = None
            # Try to get a session access token for auth header (if available)
            try:
                session = supabase.auth.session()
                if session and isinstance(session, dict):
                    token = session.get("access_token") or session.get("access_token")
            except Exception:
                token = None

            if EDGE_FUNCTION_URL:
                st.info("Calling configured EDGE_FUNCTION_URL for analysis.")
                result_resp = call_edge_function(document_id=fresh_doc.get("id"), entity_type=entity_type, query=custom_query, token=token)
                # If edge function returns success flag and analysisId etc, show it
                st.success("Edge function completed.")
                st.json(result_resp)
                # Update analysis record with returned info
                try:
                    update_analysis_record(analysis_id, {
                        "ai_analysis": result_resp,
                        "analysis_status": "completed",
                        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    })
                except Exception as e:
                    st.warning(f"Failed to update analysis record: {e}")
            else:
                # fallback to direct Gemini call if API key present
                if GEMINI_API_KEY:
                    st.info("No EDGE_FUNCTION_URL set — using GEMINI_API_KEY fallback (direct call).")
                    # attempt a simple "regulatory circulars" fetch placeholder empty list
                    comprehensive_regulatory_data = []
                    # call gemini direct
                    parsed = call_gemini_direct(fresh_doc.get("extracted_content", "")[:200000], fresh_doc.get("file_name"), entity_type, fresh_doc, comprehensive_regulatory_data)
                    st.success("Direct Gemini analysis completed.")
                    st.json(parsed)
                    update_analysis_record(analysis_id, {
                        "ai_analysis": {"raw_response": parsed},
                        "structured_output": parsed if isinstance(parsed, dict) else {},
                        "analysis_status": "completed",
                        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    })
                else:
                    raise Exception("No EDGE_FUNCTION_URL and no GEMINI_API_KEY configured. Configure one to enable analysis.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            # mark analysis failed
            try:
                update_analysis_record(analysis_id, {"analysis_status": "failed", "error": str(e)})
            except Exception:
                pass

        # cleanup local session state
        st.session_state.pop("current_analysis_id", None)
        st.session_state.pop("current_document", None)
        st.button("Refresh")
else:
    st.info("Please sign in on the left panel to upload and analyze DRHP documents.")

