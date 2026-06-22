mod auth;
mod lobby;
mod ws;

use std::sync::{Arc, Mutex};

use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};

use auth::{Db, SessionMap};
use lobby::{LobbyState, SharedLobby};

#[derive(Clone)]
struct AppState {
    db: Db,
    sessions: SessionMap,
    lobby: SharedLobby,
}

async fn ws_route(
    ws: WebSocketUpgrade,
    State(app): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| {
        ws::handle(socket, app.db, app.sessions, app.lobby)
    })
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("NOVACANA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7878);

    let db_path = std::env::var("NOVACANA_DB").unwrap_or_else(|_| "novacana.db".to_string());
    let conn = auth::open_db(&db_path).expect("failed to open database");

    let state = AppState {
        db: Arc::new(Mutex::new(conn)),
        sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
        lobby: Arc::new(Mutex::new(LobbyState::new())),
    };

    let app = Router::new()
        .route("/ws", get(ws_route))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    println!("novacana-server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind failed");
    axum::serve(listener, app).await.expect("serve failed");
}
