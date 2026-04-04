# Security Vulnerability Analysis - memory-decay-core

**Date:** 2026-04-05
**Analyzer:** security-reviewer
**Project:** memory-decay-core

---

## Summary

This analysis identifies multiple security vulnerabilities in the memory-decay-core project. The most critical issues are **arbitrary code execution** via dynamically loaded Python modules and **missing authentication** on all API endpoints including admin routes.

---

## Vulnerability Findings

### 1. Arbitrary Code Execution via `decay_fn.py` [CRITICAL]

**Location:** `src/memory_decay/server.py:51-57`

**Description:**
The `_load_best_experiment()` function dynamically loads and executes Python code from a file path that can be influenced by environment variables.

```python
decay_fn_path = experiment_dir / "decay_fn.py"
if decay_fn_path.exists():
    spec = importlib.util.spec_from_file_location("best_decay_fn", decay_fn_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # <-- Arbitrary code execution
```

**Risk:**
- If an attacker can control `experiment_dir` (via `MD_EXPERIMENT_DIR` env var) or the contents of `decay_fn.py`, they can execute arbitrary Python code on the server.
- This is particularly dangerous in multi-tenant or shared environments.

**Recommendation:**
- Validate the experiment directory path to prevent path traversal
- Use a sandboxed/isolated execution environment for custom decay functions
- Consider removing this feature or making it opt-in with clear warnings
- Add integrity checking (e.g., code signing) for custom decay functions

---

### 2. Missing Authentication on All Endpoints [CRITICAL]

**Location:** `src/memory_decay/server.py` (entire file)

**Description:**
All API endpoints, including sensitive admin endpoints, have no authentication or authorization checks:

| Endpoint | Risk |
|----------|------|
| `POST /store` | Mass memory injection |
| `POST /store-batch` | Mass memory injection |
| `DELETE /forget/{memory_id}` | Memory deletion |
| `POST /reset` | Complete data destruction |
| `GET /admin/memories` | Data exfiltration |
| `PUT /admin/decay-params` | System configuration tampering |
| `PUT /admin/tick-interval` | DoS via tick manipulation |

**Recommendation:**
- Implement authentication (JWT, API keys, etc.)
- Add role-based access control for admin endpoints
- Consider adding a secret token for admin operations

---

### 3. Path Traversal via `db_path` [HIGH]

**Location:** `src/memory_decay/server.py:255`

```python
resolved_db_path = str(Path(db_path).expanduser())
```

**Description:**
The `expanduser()` call resolves `~` to the user's home directory, but no validation prevents path traversal (e.g., `../../etc/passwd`).

**Risk:**
- An attacker controlling `MD_DB_PATH` could overwrite or read arbitrary files.

**Recommendation:**
- Validate that the resolved path is within an allowed directory
- Use `Path.resolve()` only after validation
- Add allowlist checking for database file locations

---

### 4. CORS Misconfiguration [MEDIUM]

**Location:** `src/memory_decay/server.py:304`

```python
allow_origin_regex=r"http://localhost:\d+"
```

**Description:**
CORS allows any localhost port, which may be more permissive than intended.

**Recommendation:**
- Tighten CORS regex to specific expected ports if known
- Add explicit allowed origins list
- Consider removing CORS if not needed for cross-origin requests

---

### 5. No Rate Limiting [MEDIUM]

**Location:** All endpoints

**Description:**
No rate limiting is implemented, enabling DoS attacks against the embedding API (which may have cost implications) and the server itself.

**Recommendation:**
- Add rate limiting middleware (e.g., `slowapi`)
- Implement per-user quotas
- Add request throttling for expensive operations (embedding calls)

---

### 6. Information Disclosure via Error Messages [LOW]

**Location:** Various endpoints

**Description:**
Error responses may leak internal system details:
- Line 477: `f"Memory {memory_id} not found"` - reveals internal ID structure
- Error traces in 500 responses expose stack traces

**Recommendation:**
- Use generic error messages in production
- Implement proper error handling that sanitizes error output

---

### 7. NoSQL/No Input Validation on Optional Fields [LOW]

**Location:** `src/memory_decay/server.py:68-76`

```python
class StoreRequest(BaseModel):
    text: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    mtype: str = "fact"
    category: str = ""
    associations: Optional[List[str]] = None
    created_tick: Optional[int] = None
    speaker: Optional[str] = None
```

**Description:**
While FastAPI/Pydantic provides basic validation, the `category`, `mtype`, and `speaker` fields accept arbitrary strings with no length limits or content validation.

**Recommendation:**
- Add length limits to string fields
- Validate `category` and `mtype` against allowed values/enum
- Sanitize inputs before storing or displaying

---

## Security Best Practices Checklist

- [ ] **Remove or sandbox dynamic code execution** (`decay_fn.py` loading)
- [ ] **Implement authentication** on all endpoints
- [ ] **Protect admin endpoints** with role-based access control
- [ ] **Validate and sanitize** all user inputs
- [ ] **Add rate limiting** to prevent abuse
- [ ] **Fix path traversal** in database path resolution
- [ ] **Tighten CORS configuration**
- [ ] **Use generic error messages** in production
- [ ] **Add security headers** (HSTS, X-Content-Type-Options, etc.)
- [ ] **Keep dependencies updated** and monitor for CVEs

---

## Risk Matrix

| Vulnerability | Severity | Exploitability | Impact |
|--------------|----------|----------------|--------|
| Arbitrary Code Execution | CRITICAL | Medium | Full system compromise |
| Missing Authentication | CRITICAL | Easy | Data breach, system takeover |
| Path Traversal | HIGH | Medium | File read/write |
| CORS Misconfiguration | MEDIUM | Easy | Cross-site data theft |
| No Rate Limiting | MEDIUM | Easy | DoS, cost escalation |
| Information Disclosure | LOW | Easy | Intelligence gathering |

---

## Conclusion

The most urgent issues to address are the **arbitrary code execution vulnerability** and the **complete lack of authentication**. These could allow an attacker to take full control of the server and access all stored data. Immediate remediation is recommended before any production deployment.
