use crate::game::{GameEvent, GameResult};
use crate::human::{HumanPrompt, HumanResponse};

/// Messages sent from the server to a connected client.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ServerMsg {
    // Auth
    AuthOk { username: String },
    AuthErr { reason: String },
    // Lobby
    LobbyInfo { online: u32 },
    QueueJoined { size: u8 },
    QueueLeft,
    GameFound { players: Vec<String> },
    // In-game
    Event(GameEvent),
    Prompt(HumanPrompt),
    GameEnded { result: GameResult },
}

/// Messages sent from a connected client to the server.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ClientMsg {
    // Auth
    Login { username: String, password: String },
    Register { username: String, password: String },
    // Lobby
    JoinQueue { size: u8 },
    LeaveQueue,
    // In-game
    Action(HumanResponse),
}
