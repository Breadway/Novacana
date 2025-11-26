
export enum CardType {
  Star = 'Star',
  Discovery = 'Discovery',
  Flare = 'Flare',
  Fracture = 'Fracture',
  Omen = 'Omen',
}

export enum Phase {
  Setup = 'Setup',
  Draw = 'Draw',
  Action = 'Action',
  Reaction = 'Reaction', // The Chain
  End = 'End', // Discard phase
  Victory = 'Victory',
}

export interface CardDefinition {
  name: string;
  type: CardType; // Using string from JSON, mapped to Enum
  subType?: string;
  classValue?: number;
  quantity: number;
  effectText: string;
}

export interface CardInstance extends Omit<CardDefinition, 'quantity'> {
  id: string; // Unique ID for the instance
  ownerId: number;
  targetId?: string; // ID of the card this card targets (e.g., for Omen/Athena)
}

export interface BoardCard extends CardInstance {
  attachments: CardInstance[];
}

export interface PlayerState {
  id: number;
  name: string;
  hand: CardInstance[];
  board: BoardCard[]; // Star cards played
  systemsCompleted: number;
  handRevealedUntilTurn: number;
}

export interface PendingDecision {
  type: 'CONFIRM_DEPLOYMENT' | 'PROTO_STAR_CHOICE' | 'OFFER_COMPLETE_SYSTEM' | 'SELECT_VOID_CLASS' | 'SELECT_DESTRUCTION_TARGET';
  cardId: string; // The card initiating the decision
  targetId?: string; // Optional context (where it's being played)
  compatibleDiscoveries?: string[]; // IDs of discoveries in hand that can be attached
  candidates?: CardInstance[];
}

export interface GameState {
  deck: CardInstance[];
  discardPile: CardInstance[];
  players: {
    1: PlayerState;
    2: PlayerState;
  };
  activePlayerId: 1 | 2;
  turnCount: number;
  phase: Phase;
  chainStack: CardInstance[]; // For the LIFO chain
  pendingChain: boolean; // If we are waiting for a reaction
  selectionMode: 'NONE' | 'SELECT_STAR_FOR_DISCOVERY' | 'DISCARD_TO_FIVE' | 'SELECT_TARGET_OMEN' | 'SELECT_TARGET_DESTRUCTION';
  selectedCardId: string | null; // The card currently being played/operated on
  pendingDecision: PendingDecision | null; // Modal choice state
  logs: string[];
  winner: number | null;
  victoryReason: string | null;
  isResolving: boolean; // New: For visual chain resolution
  resolvingCardId: string | null; // New: Currently resolving card for highlighting
  forbiddenStarClass: number | null;
}
