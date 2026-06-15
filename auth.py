import sqlite3
import bcrypt
from datetime import datetime

# ── Database setup ───────────────────────────────────────────
DB_PATH = "users.db"

def init_db():
    """Create the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# ── Password hashing ──────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    # Generate salt and hash
    salt = bcrypt.gensalt(rounds = 12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Check if a password matches the hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ── User operations ──────────────────────────────────────────
def signup_user(email: str, name: str, password: str) -> tuple:
    """Create a new user. Returns (success, message)."""
    # Validate inputs
    if not email or not name or not password:
        return False, "All fields are required"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if "@" not in email or "." not in email:
        return False, "Invalid email format"

    # Hash the password
    password_hash = hash_password(password)

    # Insert into database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (email, name, password_hash, created_at)
            VALUES (?, ?, ?, ?)
        """, (email.lower().strip(), name.strip(), password_hash, datetime.now().isoformat()))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, f"Account created! User ID: {user_id}"
    except sqlite3.IntegrityError:
        return False, "Email already registered"
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(email: str, password: str) -> tuple:
    """Verify credentials. Returns (success, user_data_or_message)."""
    if not email or not password:
        return False, "Email and password required"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, email, name, password_hash FROM users WHERE email = ?",
            (email.lower().strip(),)
        )
        user = cursor.fetchone()
        conn.close()

        if not user:
            return False, "Invalid email or password"

        user_id, user_email, user_name, password_hash = user

        if verify_password(password, password_hash):
            return True, {
                "id": user_id,
                "email": user_email,
                "name": user_name
            }
        else:
            return False, "Invalid email or password"

    except Exception as e:
        return False, f"Error: {str(e)}"

# ── Test the auth system ─────────────────────────────────────
if __name__ == "__main__":
    init_db()

    print("\n--- TEST 1: Signup ---")
    success, msg = signup_user("ayush@test.com", "Ayush Ladha", "mypass123")
    print(f"Signup: {success} — {msg}")

    print("\n--- TEST 2: Signup with same email (should fail) ---")
    success, msg = signup_user("ayush@test.com", "Ayush Again", "anotherpass")
    print(f"Signup: {success} — {msg}")

    print("\n--- TEST 3: Login with correct password ---")
    success, result = login_user("ayush@test.com", "mypass123")
    print(f"Login: {success} — {result}")

    print("\n--- TEST 4: Login with wrong password ---")
    success, result = login_user("ayush@test.com", "wrongpass")
    print(f"Login: {success} — {result}")

    print("\n--- TEST 5: Login with non-existent email ---")
    success, result = login_user("notreal@test.com", "anypass")
    print(f"Login: {success} — {result}")