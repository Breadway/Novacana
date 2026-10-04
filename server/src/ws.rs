use std::sync::mpsc::{self, SyncSender};

use axum::extract::ws::{Message, WebSocket};
use serde_json;

use novacana_engine::human::HumanResponse;
use novacana_engine::protocol::{ClientMsg, ServerMsg};

use crate::auth::{self, Db, SessionMap};
use crate::lobby::{self, PlayerHandle, SharedLobby};

enum ConnState {
    Unauthenticated,
    InLobby { username: String },
    InQueue { username: String, size: u8 },
    InGame { username: String, response_tx: SyncSender<HumanResponse> },
}

/// Main WebSocket connection handler.
pub async fn handle(mut socket: WebSocket, db: Db, sessions: SessionMap, lobby: SharedLobby) {
    let mut state = ConnState::Unauthenticated;
    // Channel from game thread → this WS task.
    let (game_tx, mut game_rx) = mpsc::sync_channel::<ServerMsg>(64);

    loop {
        tokio::select! {
            // Outbound: relay game/lobby messages to the client.
            msg = recv_game(&mut game_rx) => {
                match msg {
                    Some(server_msg) => {
                        let json = match serde_json::to_string(&server_msg) {
                            Ok(s) => s,
                            Err(_) => continue,
                        };
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                        // If the game ended, return to lobby state.
                        if matches!(server_msg, ServerMsg::GameEnded { .. }) {
                            if let ConnState::InGame { username, .. } = &state {
                                state = ConnState::InLobby { username: username.clone() };
                                send_lobby_info(&mut socket, &lobby).await;
                            }
                        }
                    }
                    None => break,
                }
            }
            // Inbound: receive a message from the client.
            frame = socket.recv() => {
                match frame {
                    Some(Ok(Message::Text(text))) => {
                        let client_msg: ClientMsg = match serde_json::from_str(&text) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        handle_client_msg(
                            client_msg,
                            &mut state,
                            &mut socket,
                            &db,
                            &sessions,
                            &lobby,
                            &game_tx,
                        ).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => continue,
                }
            }
        }
    }

    // Clean up on disconnect.
    on_disconnect(&state, &sessions, &lobby);
}

async fn handle_client_msg(
    msg: ClientMsg,
    state: &mut ConnState,
    socket: &mut WebSocket,
    db: &Db,
    sessions: &SessionMap,
    lobby: &SharedLobby,
    game_tx: &SyncSender<ServerMsg>,
) {
    match (&*state, msg) {
        // ── Auth ──────────────────────────────────────────────────────────────
        (ConnState::Unauthenticated, ClientMsg::Login { username, password }) => {
            match auth::login(db, sessions, &username, &password) {
                Ok(token) => {
                    let username = username.to_lowercase();
                    send_msg(socket, ServerMsg::AuthOk { username: username.clone() }).await;
                    *state = ConnState::InLobby { username };
                    send_lobby_info(socket, lobby).await;
                    // Store token for potential future use (ignored for now).
                    let _ = token;
                }
                Err(e) => send_msg(socket, ServerMsg::AuthErr { reason: e.to_string() }).await,
            }
        }

        (ConnState::Unauthenticated, ClientMsg::Register { username, password }) => {
            match auth::register(db, &username, &password) {
                Ok(()) => match auth::login(db, sessions, &username, &password) {
                    Ok(token) => {
                        let username = username.to_lowercase();
                        send_msg(socket, ServerMsg::AuthOk { username: username.clone() }).await;
                        *state = ConnState::InLobby { username };
                        send_lobby_info(socket, lobby).await;
                        let _ = token;
                    }
                    Err(e) => send_msg(socket, ServerMsg::AuthErr { reason: e.to_string() }).await,
                },
                Err(e) => send_msg(socket, ServerMsg::AuthErr { reason: e.to_string() }).await,
            }
        }

        // ── Queue management ──────────────────────────────────────────────────
        (ConnState::InLobby { username }, ClientMsg::JoinQueue { size }) if matches!(size, 2 | 3 | 4) => {
            let username = username.clone();

            let handle = PlayerHandle::new(username.clone(), game_tx.clone());
            let response_tx = handle.response_tx.clone();

            let ready = lobby.lock().unwrap().join_queue(size, handle);
            *state = ConnState::InQueue { username: username.clone(), size };
            send_msg(socket, ServerMsg::QueueJoined { size }).await;

            if let Some(players) = ready {
                // Game is ready; upgrade our state before spawning.
                *state = ConnState::InGame { username, response_tx };
                lobby::spawn_game(players);
            }
        }

        (ConnState::InQueue { username, .. }, ClientMsg::LeaveQueue) => {
            let username = username.clone();
            lobby.lock().unwrap().leave_queue(&username);
            *state = ConnState::InLobby { username };
            send_msg(socket, ServerMsg::QueueLeft).await;
        }

        // ── In-game actions ───────────────────────────────────────────────────
        (ConnState::InGame { response_tx, .. }, ClientMsg::Action(response)) => {
            let _ = response_tx.send(response);
        }

        // Invalid messages for current state - silently ignore.
        _ => {}
    }
}

fn on_disconnect(state: &ConnState, sessions: &SessionMap, lobby: &SharedLobby) {
    match state {
        ConnState::InLobby { username } | ConnState::InQueue { username, .. } => {
            lobby.lock().unwrap().leave_queue(username);
            auth::remove_token(sessions, username);
        }
        ConnState::InGame { username, .. } => {
            // Dropping response_tx causes game engine recv() to return Err,
            // which ends the game and sends GameEnded to all players.
            auth::remove_token(sessions, username);
        }
        ConnState::Unauthenticated => {}
    }
}

async fn send_lobby_info(socket: &mut WebSocket, lobby: &SharedLobby) {
    let online = lobby.lock().unwrap().online;
    send_msg(socket, ServerMsg::LobbyInfo { online }).await;
}

async fn send_msg(socket: &mut WebSocket, msg: ServerMsg) {
    if let Ok(json) = serde_json::to_string(&msg) {
        let _ = socket.send(Message::Text(json.into())).await;
    }
}

// Poll the mpsc receiver without blocking the async task.
async fn recv_game(rx: &mut mpsc::Receiver<ServerMsg>) -> Option<ServerMsg> {
    tokio::task::block_in_place(|| rx.recv().ok())
}
