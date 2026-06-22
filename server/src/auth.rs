use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use rusqlite::{Connection, params};
use uuid::Uuid;

pub type Db = Arc<Mutex<Connection>>;
pub type SessionMap = Arc<Mutex<HashMap<String, String>>>;

#[derive(Debug)]
pub enum AuthErr {
    UsernameTaken,
    InvalidUsername,
    WrongPassword,
    UserNotFound,
    Internal(String),
}

impl std::fmt::Display for AuthErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthErr::UsernameTaken => write!(f, "username already taken"),
            AuthErr::InvalidUsername => write!(f, "username must be 3-20 chars, lowercase letters/digits/underscores only"),
            AuthErr::WrongPassword => write!(f, "wrong password"),
            AuthErr::UserNotFound => write!(f, "user not found"),
            AuthErr::Internal(e) => write!(f, "internal error: {e}"),
        }
    }
}

fn validate_username(name: &str) -> bool {
    let len = name.len();
    len >= 3 && len <= 20 && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

pub fn open_db(path: &str) -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS users (
            username     TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at   INTEGER NOT NULL
        );",
    )?;
    Ok(conn)
}

pub fn register(db: &Db, username: &str, password: &str) -> Result<(), AuthErr> {
    let username = username.to_lowercase();
    if !validate_username(&username) {
        return Err(AuthErr::InvalidUsername);
    }

    let salt = SaltString::generate(&mut OsRng);
    let hash = Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| AuthErr::Internal(e.to_string()))?
        .to_string();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let db = db.lock().unwrap();
    match db.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?1, ?2, ?3)",
        params![username, hash, now],
    ) {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(e, _))
            if e.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Err(AuthErr::UsernameTaken)
        }
        Err(e) => Err(AuthErr::Internal(e.to_string())),
    }
}

pub fn login(db: &Db, sessions: &SessionMap, username: &str, password: &str) -> Result<String, AuthErr> {
    let username = username.to_lowercase();

    let stored_hash: String = {
        let db = db.lock().unwrap();
        db.query_row(
            "SELECT password_hash FROM users WHERE username = ?1",
            params![username],
            |row| row.get(0),
        )
        .map_err(|_| AuthErr::UserNotFound)?
    };

    let parsed = PasswordHash::new(&stored_hash)
        .map_err(|e| AuthErr::Internal(e.to_string()))?;
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .map_err(|_| AuthErr::WrongPassword)?;

    let token = Uuid::new_v4().to_string();
    sessions.lock().unwrap().insert(token.clone(), username);
    Ok(token)
}

pub fn resolve_token(sessions: &SessionMap, token: &str) -> Option<String> {
    sessions.lock().unwrap().get(token).cloned()
}

pub fn remove_token(sessions: &SessionMap, token: &str) {
    sessions.lock().unwrap().remove(token);
}
