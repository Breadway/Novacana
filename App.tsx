import React, { useState, useEffect, useRef } from 'react';
import { Peer } from 'peerjs';
import { 
  CardType, 
  Phase, 
  GameState, 
  PlayerState, 
  CardInstance, 
  BoardCard, 
  PendingDecision 
} from './types';
import { RAW_CARD_LIBRARY } from './constants';

// --- Helper Functions ---

const generateDeck = (): CardInstance[] => {
  let idCounter = 0;
  const deck: CardInstance[] = [];
  
  RAW_CARD_LIBRARY.forEach((def) => {
    for (let i = 0; i < def.quantity; i++) {
      deck.push({
        id: `card-${idCounter++}-${def.name.replace(/\s/g, '')}`,
        name: def.name,
        type: def.type,
        subType: def.subType,
        classValue: def.classValue,
        effectText: def.effectText,
        ownerId: 0
      });
    }
  });
  
  return shuffle(deck);
};

const shuffle = (array: CardInstance[]): CardInstance[] => {
  const newArray = [...array];
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
  }
  return newArray;
};

const initialState: GameState = {
  deck: [],
  discardPile: [],
  players: {
    1: { id: 1, name: "Player 1", hand: [], board: [], systemsCompleted: 0, handRevealedUntilTurn: 0 },
    2: { id: 2, name: "Player 2", hand: [], board: [], systemsCompleted: 0, handRevealedUntilTurn: 0 }
  },
  activePlayerId: 1,
  turnCount: 1,
  phase: Phase.Setup,
  chainStack: [],
  pendingChain: false,
  selectionMode: 'NONE',
  selectedCardId: null,
  pendingDecision: null,
  logs: ["Welcome to Novacana."],
  winner: null,
  victoryReason: null,
  isResolving: false,
  resolvingCardId: null,
  forbiddenStarClass: null
};

// --- Card Component ---

interface CardProps {
  card: CardInstance | BoardCard;
  onClick?: () => void;
  location: 'hand' | 'board' | 'discard' | 'deck' | 'chain' | 'opponent';
  isHovered?: boolean;
  onHover?: (c: CardInstance | null) => void;
  onDragStart?: (e: React.DragEvent, card: CardInstance) => void;
  onDrop?: (e: React.DragEvent, targetCard: BoardCard) => void;
  isDragging?: boolean;
  isResolving?: boolean;
  isValidDropTarget?: boolean;
}

const Card: React.FC<CardProps> = ({ 
  card, 
  onClick, 
  location, 
  isHovered, 
  onHover,
  onDragStart,
  onDrop,
  isDragging,
  isResolving,
  isValidDropTarget
}) => {
  const getTypeStyles = (type: string) => {
    switch (type) {
      case CardType.Star: return "bg-gradient-to-br from-red-950 to-orange-900 border-orange-600 text-orange-100";
      case CardType.Discovery: return "bg-gradient-to-br from-emerald-950 to-green-900 border-green-600 text-green-100";
      case CardType.Flare: return "bg-gradient-to-br from-cyan-950 to-blue-800 border-cyan-500 text-cyan-100";
      case CardType.Fracture: return "bg-gradient-to-br from-purple-950 to-gray-900 border-purple-600 text-purple-200";
      case CardType.Omen: return "bg-gradient-to-br from-indigo-950 to-blue-950 border-indigo-500 text-indigo-100";
      default: return "bg-gray-800 border-gray-600";
    }
  };

  const isBoard = location === 'board' || location === 'opponent';
  const attachments = isBoard ? (card as BoardCard).attachments || [] : [];
  
  const sizeClasses = isBoard ? 'w-40 h-56' : (location === 'chain' && isResolving) ? 'w-56 h-80' : 'w-32 h-48'; 
  
  const handleDragOver = (e: React.DragEvent) => {
    if (onDrop) {
      e.preventDefault(); 
    }
  };

  const handleDropLocal = (e: React.DragEvent) => {
    if (onDrop) {
      e.preventDefault();
      e.stopPropagation();
      onDrop(e, card as BoardCard);
    }
  };

  return (
    <div 
      onClick={onClick}
      draggable={!!onDragStart}
      onDragStart={(e) => onDragStart && onDragStart(e, card)}
      onDragOver={handleDragOver}
      onDrop={handleDropLocal}
      onMouseEnter={() => onHover && onHover(card)}
      onMouseLeave={() => onHover && onHover(null)}
      className={`
        relative select-none cursor-pointer flex-none
        ${sizeClasses}
        ${getTypeStyles(card.type)}
        border-2 rounded-xl flex flex-col p-3 shadow-xl overflow-hidden
        ${isHovered ? 'z-40 ring-4 ring-yellow-400 shadow-yellow-500/50' : 'z-10'}
        ${isDragging ? 'opacity-40 grayscale border-dashed' : ''}
        ${isResolving ? 'ring-4 ring-white z-50 shadow-[0_0_50px_rgba(255,255,255,0.8)] scale-110 rotate-0' : ''}
        ${isValidDropTarget ? 'ring-4 ring-green-400 shadow-[0_0_20px_rgba(74,222,128,0.5)]' : ''}
        transition-all duration-500 ease-out
      `}
    >
      <div className="text-sm font-bold leading-tight border-b border-white/20 pb-2 mb-2 min-h-[3em] flex items-center">
        {card.name}
      </div>
      
      <div className="flex-1 flex flex-col items-center relative">
        <div className="text-[10px] opacity-70 uppercase tracking-wider mb-1">{card.type}</div>
        
        {card.classValue && (
          <div className="flex flex-col items-center mb-1">
            <span className="text-4xl text-yellow-500 drop-shadow-lg">★</span>
            <span className="font-bold text-lg">{card.classValue}</span>
          </div>
        )}
        
        {card.subType === 'Supernova' && (
          <div className="absolute inset-0 flex items-center justify-center opacity-30">
            <div className="w-20 h-20 bg-yellow-500 rounded-full blur-xl"></div>
          </div>
        )}

        <div className="mt-auto w-full bg-black/30 p-1.5 rounded text-[10px] text-center leading-snug">
          {card.effectText}
        </div>
      </div>

      {isBoard && attachments.length > 0 && (
        <div className="absolute -bottom-3 -right-2 flex space-x-[-12px] pb-1 pr-1 pointer-events-none">
          {attachments.map((att, idx) => (
            <div 
              key={idx + att.id}
              className={`
                w-10 h-10 rounded-full border-2 border-white shadow-md flex items-center justify-center text-[12px] z-10 
                ${att.name === "Solar Eclipse" ? "bg-indigo-600 text-white shadow-[0_0_10px_indigo]" : "bg-green-700 text-white"}
                ${att.name === "Rocky Planet" ? "bg-amber-700" : ""}
                ${att.name === "Gas Giant" ? "bg-purple-700" : ""}
                ${att.name === "Planet of Life" ? "bg-blue-400 text-black font-bold" : ""}
              `}
              title={att.name}
            >
              {att.name === "Solar Eclipse" ? "☾" : att.name.charAt(0)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// --- Main App ---

export default function App() {
  const [game, setGame] = useState<GameState>(initialState);
  
  // UI States
  const [hoveredCard, setHoveredCard] = useState<CardInstance | null>(null);
  const [draggedCard, setDraggedCard] = useState<CardInstance | null>(null);
  const [dragOverBoard, setDragOverBoard] = useState(false);
  const [dragOverDeck, setDragOverDeck] = useState(false);
  const [dragOverDiscard, setDragOverDiscard] = useState(false);
  const [showVoidPreview, setShowVoidPreview] = useState(false);
  
  // Network States
  const [username, setUsername] = useState("Commander");
  const [isMultiplayer, setIsMultiplayer] = useState(false);
  const [isHost, setIsHost] = useState(false);
  const [hostId, setHostId] = useState("");
  const [joinId, setJoinId] = useState("");
  const [connectionStatus, setConnectionStatus] = useState("Offline");
  
  const peerRef = useRef<any>(null);
  const connRef = useRef<any>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [game.logs]);

  // Sync Game State (Network)
  useEffect(() => {
    if (isMultiplayer && isHost && connRef.current) {
        connRef.current.send({ type: 'UPDATE_STATE', state: game });
    }
  }, [game, isMultiplayer, isHost]);

  // Win Condition Check
  useEffect(() => {
    if (game.phase === Phase.Setup || game.winner) return;

    let winnerFound = false;
    let newPlayers = { ...game.players };

    for (const pid of [1, 2] as const) {
        const player = game.players[pid];
        let systemsCompleted = 0;

        player.board.forEach(star => {
            if (star.type === CardType.Star && star.classValue) {
                const discoveryCount = star.attachments.filter(a => a.type === CardType.Discovery).length;
                if (discoveryCount >= star.classValue) {
                    systemsCompleted++;
                }
            }
        });

        if (systemsCompleted !== player.systemsCompleted) {
             newPlayers[pid] = { ...player, systemsCompleted };
        }

        if (systemsCompleted >= 3 && !winnerFound) {
            winnerFound = true;
            setGame(prev => ({
                ...prev,
                players: newPlayers,
                winner: pid,
                victoryReason: `GALACTIC DOMINANCE: ${player.name} completed 3 Star Systems!`
            }));
        }
    }
  }, [game.players, game.winner, game.phase]);


  // --- Network Logic ---
  const initNetwork = (host: boolean) => {
      setIsMultiplayer(true);
      setIsHost(host);
      setConnectionStatus("Initializing...");
      import('peerjs').then(({ Peer }) => {
          const peer = new Peer();
          peerRef.current = peer;
          peer.on('open', (id: string) => {
              setHostId(id);
              setConnectionStatus(host ? "Waiting for Opponent..." : "Ready to Join");
          });
          if (host) {
              peer.on('connection', (conn: any) => {
                  setConnectionStatus("Connected!");
                  connRef.current = conn;
                  setupConnection(conn);
                  startGame(true);
              });
          }
      });
  };

  const joinGame = () => {
      if (!joinId || !peerRef.current) return;
      setConnectionStatus("Connecting...");
      const conn = peerRef.current.connect(joinId);
      connRef.current = conn;
      conn.on('open', () => {
          setConnectionStatus("Connected!");
          setupConnection(conn);
          startGame(false);
      });
  };

  const setupConnection = (conn: any) => {
      conn.on('data', (data: any) => {
          if (data.type === 'UPDATE_STATE') {
              if (!isHost) setGame(data.state);
          }
      });
  };

  const startGame = (asHost: boolean) => {
    const newDeck = generateDeck();
    const p1Hand = newDeck.splice(0, 5).map(c => ({ ...c, ownerId: 1 }));
    const p2Hand = newDeck.splice(0, 5).map(c => ({ ...c, ownerId: 2 }));
    
    let p1Draw = null;
    if (newDeck.length > 0) {
      p1Draw = newDeck.shift()!;
      p1Draw.ownerId = 1;
      p1Hand.push(p1Draw);
    }

    const startState: GameState = {
      ...initialState,
      deck: newDeck,
      players: {
        1: { id: 1, name: asHost ? username : "Opponent", hand: p1Hand, board: [], systemsCompleted: 0, handRevealedUntilTurn: 0 },
        2: { id: 2, name: !asHost ? username : "Opponent", hand: p2Hand, board: [], systemsCompleted: 0, handRevealedUntilTurn: 0 }
      },
      phase: Phase.Action,
      logs: ["Game Initialized.", p1Draw ? `Player 1 drew ${p1Draw.name}.` : "Deck empty."]
    };

    if (asHost || !isMultiplayer) setGame(startState);
  };

  const startLocalGame = () => {
      setIsMultiplayer(false);
      startGame(true); 
      setGame(prev => ({
          ...prev,
          players: {
              ...prev.players,
              1: { ...prev.players[1], name: username || "Player 1" },
              2: { ...prev.players[2], name: "Player 2" }
          }
      }));
  };

  // --- Game Loop Hooks (Chain) ---
  useEffect(() => {
    if (game.isResolving && game.chainStack.length > 0) {
      const topCard = game.chainStack[game.chainStack.length - 1];
      if (game.resolvingCardId !== topCard.id) {
          setGame(prev => ({ ...prev, resolvingCardId: topCard.id }));
          return;
      }
      const timer = setTimeout(() => resolveNextChainStep(), 2500); 
      return () => clearTimeout(timer);

    } else if (game.isResolving && game.chainStack.length === 0) {
      const timer = setTimeout(() => {
        setGame(prev => ({
          ...prev,
          isResolving: false,
          resolvingCardId: null,
          phase: Phase.Action,
          pendingChain: false,
          logs: [...prev.logs, "--- Chain Empty. Action Phase resumes. ---"]
        }));
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [game.isResolving, game.chainStack, game.resolvingCardId]);

  const resolveNextChainStep = () => {
    setGame(prev => {
      const newState = { ...prev };
      const stack = [...newState.chainStack];
      const currentCard = stack.pop();
      if (!currentCard) return prev;

      newState.chainStack = stack;
      newState.resolvingCardId = null;
      newState.discardPile = [currentCard, ...newState.discardPile];

      let logMsg = `Resolved: ${currentCard.name}`;
      const owner = newState.players[currentCard.ownerId];
      const opponent = newState.players[currentCard.ownerId === 1 ? 2 : 1];
      let shouldRecycle = true;

      switch (currentCard.name) {
          case "Cosmic Denial":
              if (newState.chainStack.length > 0) {
                  const negated = newState.chainStack.pop();
                  newState.discardPile.unshift(negated!);
                  logMsg = `Cosmic Denial NEGATED ${negated!.name}!`;
              }
              break;
          case "Solar Eclipse":
              if (currentCard.targetId) {
                  const target = owner.board.find(b => b.id === currentCard.targetId);
                  if (target) {
                      target.attachments.push(currentCard);
                      newState.discardPile.shift(); 
                      shouldRecycle = false;
                      logMsg = "Solar Eclipse attached.";
                  }
              }
              break;
          case "Planet of Life":
               if (currentCard.targetId) {
                  const targetStar = owner.board.find(b => b.id === currentCard.targetId);
                  if (targetStar && targetStar.attachments.some(a => a.name === "Rocky Planet")) {
                      targetStar.attachments.push(currentCard);
                      newState.discardPile.shift();
                      shouldRecycle = false;
                      logMsg = `Planet of Life began on ${targetStar.name}.`;
                  }
               }
               break;
          case "Athena":
               if (currentCard.targetId) {
                 let found = false;
                 opponent.board.forEach(star => {
                   if (star.attachments.some(a => a.name === "Solar Eclipse")) return;
                   const idx = star.attachments.findIndex(a => a.id === currentCard.targetId);
                   if (idx > -1) {
                     const destroyed = star.attachments.splice(idx, 1)[0];
                     newState.discardPile.unshift(destroyed);
                     logMsg = `Athena destroyed ${destroyed.name}!`;
                     found = true;
                   }
                 });
               }
               break;
          case "Ignition":
               if (currentCard.targetId) {
                   const targetStar = opponent.board.find(s => s.id === currentCard.targetId);
                   if (targetStar && !targetStar.attachments.some(a => a.name === "Solar Eclipse")) {
                       const gasIdx = targetStar.attachments.findIndex(a => a.name === "Gas Giant");
                       if (gasIdx > -1) {
                           targetStar.attachments.splice(gasIdx, 1);
                           logMsg = `Ignition destroyed Gas Giant!`;
                           const deckIdx = newState.deck.findIndex(c => c.name === "Yellow Dwarf");
                           if (deckIdx > -1) {
                               const yd = newState.deck.splice(deckIdx, 1)[0];
                               yd.ownerId = currentCard.ownerId;
                               owner.hand.push(yd);
                           }
                       }
                   }
               }
               break;
          case "Stellar Alignment":
                const draws = newState.deck.splice(0, 2);
                draws.forEach(c => { c.ownerId = currentCard.ownerId; owner.hand.push(c); });
                logMsg = `Stellar Alignment: Drawn 2.`;
                break;
          case "Wormhole":
                 if (opponent.hand.length > 0) {
                   const idx = Math.floor(Math.random() * opponent.hand.length);
                   const stolen = opponent.hand.splice(idx, 1)[0];
                   stolen.ownerId = currentCard.ownerId;
                   owner.hand.push(stolen);
                   logMsg = `Wormhole stole a card!`;
                 }
                 break;
          case "Astral Projection":
                 opponent.handRevealedUntilTurn = newState.turnCount + 2;
                 logMsg = `Hand revealed for 2 turns.`;
                 break;
          case "Black Hole":
                 const allCards = [
                     ...owner.hand, ...owner.board.flatMap(s => [s, ...s.attachments]),
                     ...opponent.hand, ...opponent.board.flatMap(s => [s, ...s.attachments]),
                     ...newState.discardPile
                 ];
                 owner.hand = []; owner.board = [];
                 opponent.hand = []; opponent.board = [];
                 newState.deck = shuffle([...newState.deck, ...allCards]);
                 newState.discardPile = [];
                 newState.turnCount += 3; 
                 logMsg = `Black Hole RESET THE GAME.`;
                 for(let i=0; i<5; i++) {
                     if(newState.deck.length) owner.hand.push({ ...newState.deck.shift()!, ownerId: currentCard.ownerId });
                     if(newState.deck.length) opponent.hand.push({ ...newState.deck.shift()!, ownerId: opponentId });
                 }
                 shouldRecycle = false;
                 break;
          case "Pulsar":
                 const cardsToShuffle = [...owner.hand, ...owner.board.flatMap(s => [s, ...s.attachments])];
                 const count = cardsToShuffle.length;
                 owner.hand = []; owner.board = [];
                 newState.deck = shuffle([...newState.deck, ...cardsToShuffle]);
                 for(let i=0; i<count; i++) {
                     if(newState.deck.length) owner.hand.push({ ...newState.deck.shift()!, ownerId: currentCard.ownerId });
                 }
                 logMsg = `Pulsar: Reshuffled and drawn.`;
                 break;
          case "Dark Matter Void":
                 logMsg = `Void: Class ${newState.forbiddenStarClass} banned next turn.`;
                 break;
          case "Quantum Entanglement Failure":
                 if (owner.hand.length && opponent.hand.length) {
                     const oIdx = Math.floor(Math.random() * owner.hand.length);
                     const opIdx = Math.floor(Math.random() * opponent.hand.length);
                     const oCard = owner.hand[oIdx];
                     const opCard = opponent.hand[opIdx];
                     owner.hand[oIdx] = opCard; opponent.hand[opIdx] = oCard;
                     oCard.ownerId = opponentId; opCard.ownerId = currentCard.ownerId;
                     logMsg = "Quantum Entanglement swapped cards.";
                 }
                 break;
          case "Stellar Nursery Collapse":
                 const milled = newState.deck.splice(0, 5);
                 newState.discardPile = [...milled, ...newState.discardPile];
                 logMsg = `Milled 5 cards.`;
                 break;
          case "Supernova - Fusion":
          case "Supernova - Gravity":
          case "Supernova - Gas":
                 const supernovas = ["Supernova - Fusion", "Supernova - Gravity", "Supernova - Gas"];
                 const stackNames = newState.chainStack.map(c => c.name);
                 stackNames.push(currentCard.name);
                 if (supernovas.every(s => stackNames.includes(s))) {
                     newState.winner = currentCard.ownerId;
                     newState.victoryReason = "SUPERNOVA EVENT TRIGGERED!";
                 }
                 break;
           default:
             logMsg = `${currentCard.name} resolved.`;
      }
      
      if (shouldRecycle && newState.deck.length === 0 && newState.discardPile.length > 0) {
          newState.deck = shuffle([...newState.discardPile]);
          newState.discardPile = [];
          logMsg += " (Deck Reshuffled)";
      }

      newState.logs = [...newState.logs, logMsg];
      return newState;
    });
  };

  // --- Handlers & Logic ---

  const nextPhaseOrTurn = () => {
    setGame(prev => {
      // Chain Resolution
      if (prev.pendingChain && prev.chainStack.length > 0) {
         return { ...prev, pendingChain: false, isResolving: true, logs: [...prev.logs, "Resolving Chain..."] };
      }

      // Turn Passing
      if (prev.phase === Phase.Action) {
        const activeHand = prev.players[prev.activePlayerId].hand;
        if (activeHand.length > 5) {
          return { ...prev, selectionMode: 'DISCARD_TO_FIVE', logs: [...prev.logs, "Hand limit! Discard to 5."] };
        }
        
        const nextPlayerId = prev.activePlayerId === 1 ? 2 : 1;
        const nextPlayer = prev.players[nextPlayerId];
        const nextHand = [...nextPlayer.hand];
        let newDeck = [...prev.deck];
        let newDiscard = [...prev.discardPile];

        if (newDeck.length > 0) {
           const c = newDeck.shift()!;
           c.ownerId = nextPlayerId;
           nextHand.push(c);
        } else if (newDiscard.length > 0) {
           newDeck = shuffle(newDiscard);
           newDiscard = [];
           const c = newDeck.shift()!;
           c.ownerId = nextPlayerId;
           nextHand.push(c);
        }
        
        return {
          ...prev,
          activePlayerId: nextPlayerId,
          turnCount: prev.turnCount + 1,
          deck: newDeck,
          discardPile: newDiscard,
          forbiddenStarClass: null,
          players: { ...prev.players, [nextPlayerId]: { ...nextPlayer, hand: nextHand } },
          logs: [...prev.logs, `--- Turn Ended. Player ${nextPlayerId}'s Turn ---`]
        };
      }
      return prev;
    });
  };

  const handleDiscard = (card: CardInstance) => {
     setGame(prev => {
        const p = prev.players[prev.activePlayerId];
        const newHand = p.hand.filter(c => c.id !== card.id);
        const newDiscard = [card, ...prev.discardPile];
        let mode = prev.selectionMode;
        if (mode === 'DISCARD_TO_FIVE' && newHand.length <= 5) mode = 'NONE';
        
        return {
           ...prev,
           selectionMode: mode,
           discardPile: newDiscard,
           players: { ...prev.players, [prev.activePlayerId]: { ...p, hand: newHand } },
           logs: [...prev.logs, `Discarded ${card.name}.`]
        };
     });
  };

  // --- Drag & Drop ---
  const handleDragStart = (e: React.DragEvent, card: CardInstance) => {
    if (game.phase === Phase.Setup) return;
    if (game.phase === Phase.Action && card.type === CardType.Star && game.forbiddenStarClass === card.classValue) return;
    if (game.phase === Phase.Action && getActivePlayer().hand.length <= 5 && game.selectionMode !== 'DISCARD_TO_FIVE') return;
    if (game.isResolving || game.pendingDecision) { e.preventDefault(); return; }

    setDraggedCard(card);
    e.dataTransfer.setData('text/plain', card.id);
  };

  const handleDragOverBoard = (e: React.DragEvent) => { e.preventDefault(); setDragOverBoard(true); };
  const handleDragOverDeck = (e: React.DragEvent) => { e.preventDefault(); setDragOverDeck(true); };
  const handleDragOverDiscard = (e: React.DragEvent) => { e.preventDefault(); setDragOverDiscard(true); };

  const handleDropOnDiscard = (e: React.DragEvent) => {
      e.preventDefault(); setDragOverDiscard(false);
      if (draggedCard && (game.selectionMode === 'DISCARD_TO_FIVE' || getActivePlayer().hand.length > 5)) {
          handleDiscard(draggedCard);
      }
      setDraggedCard(null);
  };

  const handleDropOnBoard = (e: React.DragEvent) => {
    e.preventDefault(); setDragOverBoard(false);
    if (!draggedCard || game.phase === Phase.Setup) return;
    if (game.phase === Phase.Action && getActivePlayer().hand.length <= 5 && game.selectionMode !== 'DISCARD_TO_FIVE') return;

    if (draggedCard.type === CardType.Star) {
         if (game.forbiddenStarClass && draggedCard.classValue === game.forbiddenStarClass) {
             setGame(prev => ({ ...prev, logs: [...prev.logs, "Forbidden Class!"] }));
             setDraggedCard(null); return;
         }
         const player = getActivePlayer();
         const compatible = player.hand.filter(c => c.type === CardType.Discovery);
         const required = draggedCard.classValue || 99;
         
         if (compatible.length >= required && required > 0) {
            setGame(prev => ({
               ...prev,
               pendingDecision: { type: 'OFFER_COMPLETE_SYSTEM', cardId: draggedCard.id, compatibleDiscoveries: compatible.slice(0, required).map(d => d.id) }
            }));
         } else {
            playCard(draggedCard);
         }
    } else if ([CardType.Flare, CardType.Fracture, CardType.Omen].includes(draggedCard.type)) {
         const needsTarget = ["Athena", "Ignition", "Solar Eclipse", "Planet of Life"];
         if (needsTarget.includes(draggedCard.name)) {
             setGame(prev => ({ ...prev, logs: [...prev.logs, `${draggedCard.name} requires a specific target (Star or Discovery).`] }));
         } else if (draggedCard.name === "Dark Matter Void") {
             setGame(prev => ({ 
                  ...prev, 
                  pendingDecision: { type: 'SELECT_VOID_CLASS', cardId: draggedCard.id }
              }));
         } else {
             playCard(draggedCard);
         }
    }
    setDraggedCard(null);
  };

  const handleDropOnStar = (e: React.DragEvent, targetStar: BoardCard) => {
    e.stopPropagation(); setDragOverBoard(false);
    if (!draggedCard) return;

    if (targetStar.ownerId === game.activePlayerId) {
        if (draggedCard.type === CardType.Discovery) {
           const discCount = targetStar.attachments.filter(a => a.type === CardType.Discovery).length;
           if (targetStar.classValue && discCount >= targetStar.classValue) {
               setGame(prev => ({ ...prev, logs: [...prev.logs, "System Full."] }));
           } else {
               playCard(draggedCard, targetStar.id);
           }
        } else if (draggedCard.name === "Solar Eclipse") {
            const discCount = targetStar.attachments.filter(a => a.type === CardType.Discovery).length;
            if (targetStar.classValue && discCount >= targetStar.classValue) {
                playCard(draggedCard, targetStar.id);
            }
        } else if (draggedCard.name === "Planet of Life") {
            if (targetStar.attachments.some(a => a.name === "Rocky Planet")) {
                playCard(draggedCard, targetStar.id);
            }
        }
    } else {
        if (draggedCard.name === "Athena") {
             const discs = targetStar.attachments.filter(a => a.type === CardType.Discovery);
             if (discs.length > 1) {
                 setGame(prev => ({ ...prev, pendingDecision: { type: 'SELECT_DESTRUCTION_TARGET', cardId: draggedCard.id, targetId: targetStar.id, candidates: discs } }));
             } else if (discs.length === 1) {
                 playCard(draggedCard, discs[0].id);
             }
        } else if (draggedCard.name === "Ignition") {
             if (targetStar.attachments.some(a => a.name === "Gas Giant")) {
                 playCard(draggedCard, targetStar.id);
             }
        }
    }
    setDraggedCard(null);
  };

  const playCard = (card: CardInstance, targetId?: string) => {
    setGame(prev => {
       const player = prev.players[card.ownerId];
       const newHand = player.hand.filter(c => c.id !== card.id);
       const cardWithTarget = { ...card, targetId };

       if (card.type === CardType.Star) {
          if (card.name === "Proto-Star") {
             return { ...prev, pendingDecision: { type: 'PROTO_STAR_CHOICE', cardId: card.id } };
          }
          const newBoardCard: BoardCard = { ...cardWithTarget, attachments: [] };
          return {
             ...prev,
             players: { ...prev.players, [card.ownerId]: { ...player, hand: newHand, board: [...player.board, newBoardCard] } },
             logs: [...prev.logs, `Deployed ${card.name}.`]
          };
       }
       if (card.type === CardType.Discovery && targetId) {
          const newBoard = player.board.map(b => {
             if (b.id === targetId) return { ...b, attachments: [...b.attachments, cardWithTarget] };
             return b;
          });
          return {
            ...prev,
            players: { ...prev.players, [card.ownerId]: { ...player, hand: newHand, board: newBoard } },
            logs: [...prev.logs, `Attached ${card.name}.`]
          };
       }
       // Default chain play
       return {
          ...prev,
          players: { ...prev.players, [card.ownerId]: { ...player, hand: newHand } },
          chainStack: [...prev.chainStack, cardWithTarget],
          pendingChain: true,
          phase: Phase.Reaction,
          logs: [...prev.logs, `${card.name} played to Chain.`]
       };
    });
  };

  // --- Decision Handlers ---
  const handleCompleteSystem = (exec: boolean) => {
      if(!game.pendingDecision) return;
      const { cardId, compatibleDiscoveries } = game.pendingDecision;
      const player = game.players[game.activePlayerId];
      const card = player.hand.find(c => c.id === cardId);
      if (exec && card && compatibleDiscoveries) {
          const discoveries = player.hand.filter(c => compatibleDiscoveries.includes(c.id));
          const newHand = player.hand.filter(c => c.id !== cardId && !compatibleDiscoveries.includes(c.id));
          const newBoardCard: BoardCard = { ...card, attachments: discoveries };
          setGame(prev => ({
              ...prev, pendingDecision: null,
              players: { ...prev.players, [prev.activePlayerId]: { ...player, hand: newHand, board: [...player.board, newBoardCard] } },
              logs: [...prev.logs, "Deployed Complete System!"]
          }));
      } else if (card) {
          playCard(card);
          setGame(prev => ({ ...prev, pendingDecision: null }));
      }
  };

  const handleProtoChoice = (choice: 'DRAW' | 'SEARCH') => {
      setGame(prev => {
          const player = prev.players[prev.activePlayerId];
          const protoId = prev.pendingDecision?.cardId;
          const protoCard = player.hand.find(c => c.id === protoId);
          let newHand = player.hand.filter(c => c.id !== protoId);
          const newState = { ...prev, pendingDecision: null };

          if (choice === 'DRAW') {
              const drawn = newState.deck.splice(0, 2);
              drawn.forEach(c => { c.ownerId = prev.activePlayerId; newHand.push(c); });
              newState.logs = [...prev.logs, "Proto-Star: Drew 2."];
          } else {
              const idx = newState.deck.findIndex(c => c.name === "White Dwarf");
              if (idx > -1) {
                  const card = newState.deck.splice(idx, 1)[0];
                  card.ownerId = prev.activePlayerId;
                  newHand.push(card);
                  newState.logs = [...prev.logs, "Proto-Star: Found White Dwarf."];
              } else {
                  newState.logs = [...prev.logs, "Proto-Star: Search failed."];
              }
          }
          newState.players[prev.activePlayerId].hand = newHand;
          if (protoCard) newState.discardPile = [protoCard, ...newState.discardPile];
          return newState;
      });
  };

  const handleDestructionTarget = (targetId: string) => {
      if(!game.pendingDecision) return;
      const cardId = game.pendingDecision.cardId;
      const player = game.players[game.activePlayerId];
      const card = player.hand.find(c => c.id === cardId);
      if(card) {
          playCard(card, targetId);
      }
      setGame(prev => ({ ...prev, pendingDecision: null }));
  };

  const handleVoidClass = (classVal: number) => {
       if(!game.pendingDecision) return;
       const cardId = game.pendingDecision.cardId;
       const player = game.players[game.activePlayerId];
       const card = player.hand.find(c => c.id === cardId);
       setGame(prev => ({ ...prev, forbiddenStarClass: classVal, pendingDecision: null }));
       if(card) playCard(card);
  };

  const getActivePlayer = () => game.players[game.activePlayerId];
  const getOpponentPlayer = () => game.players[game.activePlayerId === 1 ? 2 : 1];

  // --- Render ---

  if (game.phase === Phase.Setup) {
      return (
          <div className="w-full h-screen bg-gray-900 text-white flex flex-col items-center justify-center relative">
              <div className="absolute inset-0 opacity-30 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')]"></div>
              <div className="z-10 bg-gray-800 p-10 rounded-2xl shadow-2xl border-2 border-yellow-500 max-w-md w-full">
                  <h1 className="text-4xl font-bold text-yellow-500 mb-2 text-center tracking-wider">NOVACANA</h1>
                  <div className="text-gray-400 text-center mb-8 text-sm">Galactic Strategy Card Game</div>
                  
                  <div className="mb-6">
                      <label className="block text-xs font-bold text-gray-500 mb-1">CALL SIGN</label>
                      <input 
                        type="text" 
                        value={username} 
                        onChange={(e) => setUsername(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded p-3 text-white focus:border-yellow-500 outline-none"
                      />
                  </div>

                  <div className="space-y-3">
                      <button onClick={startLocalGame} className="w-full py-4 bg-yellow-600 hover:bg-yellow-500 text-black font-bold rounded shadow-lg transition">
                          PLAY LOCAL (PASS & PLAY)
                      </button>
                      <div className="h-px bg-gray-700 my-4"></div>
                      <button onClick={() => initNetwork(true)} className="w-full py-3 bg-indigo-700 hover:bg-indigo-600 rounded font-bold transition">
                          HOST NETWORK GAME
                      </button>
                      <div className="flex space-x-2">
                          <input 
                            type="text" 
                            placeholder="Enter Host ID" 
                            value={joinId}
                            onChange={(e) => setJoinId(e.target.value)}
                            className="flex-1 bg-gray-900 border border-gray-700 rounded p-3 text-sm"
                          />
                          <button onClick={joinGame} className="px-6 bg-indigo-900 hover:bg-indigo-800 border border-indigo-600 rounded font-bold transition">
                              JOIN
                          </button>
                      </div>
                      {isMultiplayer && (
                          <div className="mt-4 p-3 bg-black/50 rounded text-center">
                              <div className="text-xs text-gray-500 mb-1">CONNECTION STATUS</div>
                              <div className="font-mono text-green-400">{connectionStatus}</div>
                              {isHost && hostId && (
                                  <div className="mt-2 text-xs text-gray-400 select-all cursor-pointer bg-gray-900 p-2 rounded border border-gray-700">
                                      ID: {hostId}
                                  </div>
                              )}
                          </div>
                      )}
                  </div>
              </div>
          </div>
      );
  }

  return (
    <div className="w-full h-screen bg-gray-900 text-white flex flex-col overflow-hidden font-sans relative">
      <div className="absolute inset-0 pointer-events-none opacity-20 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')]"></div>
      
      {/* Modal Overlay */}
      {game.pendingDecision && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm">
           <div className="bg-gray-800 border-2 border-yellow-500 p-6 rounded-lg shadow-xl max-w-sm w-full animate-pop-in">
               
               {game.pendingDecision.type === 'OFFER_COMPLETE_SYSTEM' && (
                   <>
                       <h3 className="text-xl font-bold text-yellow-500 mb-4">Complete System?</h3>
                       <button onClick={() => handleCompleteSystem(true)} className="w-full py-2 bg-yellow-600 mb-2 rounded font-bold hover:bg-yellow-500">Yes, Deploy Full System</button>
                       <button onClick={() => handleCompleteSystem(false)} className="w-full py-2 bg-gray-700 rounded hover:bg-gray-600">No, Star Only</button>
                   </>
               )}

               {game.pendingDecision.type === 'PROTO_STAR_CHOICE' && (
                   <>
                       <h3 className="text-xl font-bold text-yellow-500 mb-4">Proto-Star Action</h3>
                       <button onClick={() => handleProtoChoice('DRAW')} className="w-full py-2 bg-blue-600 mb-2 rounded font-bold hover:bg-blue-500">Draw 2 Cards</button>
                       <button onClick={() => handleProtoChoice('SEARCH')} className="w-full py-2 bg-purple-600 rounded font-bold hover:bg-purple-500">Search White Dwarf</button>
                   </>
               )}

               {game.pendingDecision.type === 'SELECT_VOID_CLASS' && (
                   <>
                       <h3 className="text-xl font-bold text-purple-500 mb-4">Ban Star Class</h3>
                       <p className="text-xs text-gray-400 mb-4">Opponent cannot play this Star Class next turn.</p>
                       <div className="grid grid-cols-3 gap-2">
                           {[2,3,4,5,6].map(c => (
                               <button key={c} onClick={() => handleVoidClass(c)} className="p-3 bg-gray-700 hover:bg-purple-900 rounded font-bold text-lg">{c}</button>
                           ))}
                       </div>
                   </>
               )}

               {game.pendingDecision.type === 'SELECT_DESTRUCTION_TARGET' && (
                   <>
                       <h3 className="text-xl font-bold text-red-500 mb-4">Destroy Target</h3>
                       <div className="grid grid-cols-2 gap-2">
                           {game.pendingDecision.candidates?.map(c => (
                               <button key={c.id} onClick={() => handleDestructionTarget(c.id)} className="p-2 bg-gray-700 hover:bg-red-900 border border-gray-600 rounded text-sm">{c.name}</button>
                           ))}
                       </div>
                   </>
               )}

               <button onClick={() => setGame(prev => ({...prev, pendingDecision: null}))} className="mt-6 text-xs text-gray-500 hover:text-white underline w-full text-center">Cancel</button>
           </div>
        </div>
      )}

      {/* Victory Screen */}
      {game.winner && (
          <div className="absolute inset-0 z-[200] bg-black/95 flex flex-col items-center justify-center animate-slide-up">
              <h1 className="text-6xl font-bold text-yellow-500 mb-4">VICTORY</h1>
              <p className="text-2xl text-white mb-8 text-center px-4">{game.victoryReason}</p>
              <button onClick={() => setGame(initialState)} className="px-8 py-3 bg-white text-black font-bold rounded-full hover:scale-105 transition">Return to Menu</button>
          </div>
      )}

      {/* Top Bar */}
      <div className="h-16 flex-none bg-gray-950 flex items-center justify-between px-6 border-b border-gray-800 z-10 relative">
         <div className="flex items-center space-x-4">
            <div className={`w-3 h-3 rounded-full ${game.activePlayerId === 1 ? 'bg-gray-600' : 'bg-red-500'}`}></div>
            <span className="font-bold text-lg text-gray-300">
              {getOpponentPlayer().name}
            </span>
            <span className="text-sm text-gray-500 bg-gray-900 px-2 py-1 rounded">Hand: {getOpponentPlayer().hand.length}</span>
            <span className="text-xs text-yellow-600 border border-yellow-900 px-2 py-1 rounded">Systems: {getOpponentPlayer().systemsCompleted}/3</span>
         </div>
         <div className="text-white font-bold text-xl tracking-wider text-yellow-500/80">NOVACANA</div>
      </div>

      <div className="flex-1 flex relative overflow-hidden min-h-0" 
           onDragOver={handleDragOverBoard}
           onDrop={handleDropOnBoard}
      >
        
        {/* Sidebar Left: Deck/Void */}
        <div className="w-32 bg-gray-900/50 flex flex-col justify-center items-center space-y-8 p-2 border-r border-white/5 z-20 relative flex-none">
            <div className="w-24 h-36 rounded-lg border-2 border-gray-600 flex flex-col items-center justify-center bg-gray-800">
                <div className="text-gray-400 text-xs font-bold mb-1">DECK</div>
                <div className="text-2xl font-bold">{game.deck.length}</div>
            </div>
            <div 
                className={`w-24 h-24 rounded-full border-4 border-double border-purple-900 flex flex-col items-center justify-center bg-black transition-all cursor-help relative ${dragOverDiscard ? 'scale-110 border-purple-500' : ''}`}
                onDragOver={handleDragOverDiscard}
                onDrop={handleDropOnDiscard}
                onMouseEnter={() => setShowVoidPreview(true)}
                onMouseLeave={() => setShowVoidPreview(false)}
            >
                <div className="text-purple-400 text-[10px] font-bold tracking-widest">VOID</div>
                <div className="text-xs text-purple-700 mt-1">{game.discardPile.length}</div>
                
                {/* Void Preview */}
                {showVoidPreview && (
                    <div className="absolute left-full ml-4 top-0 bg-gray-900 border border-purple-500 p-3 rounded-lg shadow-2xl z-50 w-56 max-h-80 overflow-y-auto">
                        <div className="text-xs font-bold text-purple-400 mb-2 border-b border-purple-800 pb-1">VOID CONTENT</div>
                        {game.discardPile.map((c, i) => (
                            <div key={i} className="text-xs text-gray-300 py-1 border-b border-gray-800 flex justify-between">
                                <span>{c.name}</span>
                                <span className="text-[10px] text-gray-600">{c.type[0]}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>

        {/* Board */}
        <div className="flex-1 flex flex-col relative z-0 min-h-0 min-w-0">
           {/* Opponent Board */}
           <div className="flex-1 border-b border-white/5 p-4 flex items-center justify-center space-x-4 bg-black/20 overflow-x-auto min-h-0">
              {getOpponentPlayer().board.map(star => (
                 <Card key={star.id} card={star} location="opponent" onDrop={(e) => handleDropOnStar(e, star)} onHover={setHoveredCard} />
              ))}
           </div>

           {/* Chain */}
           {game.chainStack.length > 0 && (
             <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm pointer-events-none">
                <div className="flex flex-col items-center w-full h-full relative">
                   <h2 className="text-3xl font-bold mb-4 mt-8 text-yellow-400 drop-shadow-md uppercase tracking-widest">
                     {game.isResolving ? "Resolving Chain" : "Chain Pending"}
                   </h2>
                   
                   {/* Stack Visual - Dimmed when resolving specific card */}
                   <div className="flex items-center justify-center space-x-[-80px] mb-8 perspective-1000 min-h-[120px]">
                      {game.chainStack.map((c, i) => {
                        const isCurrent = c.id === game.resolvingCardId;
                        return (
                            <div 
                                key={c.id} 
                                style={{ 
                                    transform: `translateX(${i * 30}px) translateZ(${i * 5}px) scale(${1 - (game.chainStack.length - i) * 0.05})`, 
                                    zIndex: i,
                                    opacity: game.isResolving && !isCurrent ? 0.3 : 1
                                }}
                            >
                               {/* Tiny preview cards in stack */}
                               <div className={`w-20 h-32 rounded bg-gray-700 border border-gray-500 shadow-xl`}></div>
                            </div>
                        );
                      })}
                   </div>

                   {/* ACTIVE RESOLVING CARD DISPLAY */}
                   <div className="flex-1 flex items-center justify-center w-full relative">
                        {game.isResolving && game.resolvingCardId && (
                            <div className="animate-pop-in z-50 flex flex-col items-center">
                                <Card 
                                    card={game.chainStack.find(c => c.id === game.resolvingCardId)!} 
                                    location="chain" 
                                    isResolving={true} 
                                />
                                <div className="mt-6 text-2xl font-bold text-white text-center bg-black/60 px-6 py-2 rounded-full border border-yellow-500/50">
                                    Resolving Effect...
                                </div>
                            </div>
                        )}
                        {!game.isResolving && (
                             <div className="z-50 animate-pop-in">
                                 <Card 
                                    card={game.chainStack[game.chainStack.length - 1]} 
                                    location="chain" 
                                 />
                                 <div className="mt-4 text-center text-gray-400 text-sm">Most Recent Link</div>
                             </div>
                        )}
                   </div>

                   {!game.isResolving && (
                     <div className="mb-12 pointer-events-auto">
                        <button onClick={nextPhaseOrTurn} className="bg-red-600 hover:bg-red-500 text-white font-bold py-4 px-12 rounded-full shadow-[0_0_30px_rgba(220,38,38,0.6)] transform hover:scale-105 transition-all uppercase tracking-widest text-lg border-2 border-red-400">
                            Resolve Chain
                        </button>
                     </div>
                   )}
                </div>
             </div>
           )}

           {/* Player Board */}
           <div 
             className={`flex-1 p-4 flex items-center justify-center space-x-6 transition-colors duration-300 relative overflow-x-auto min-h-0 ${dragOverBoard ? 'bg-indigo-900/30' : ''}`}
           >
              {getActivePlayer().board.length === 0 && !draggedCard && (
                  <div className="text-gray-500/30 border-2 border-dashed border-gray-700/50 rounded-xl p-10 select-none pointer-events-none">
                      Deploy Stars or Effects Here
                  </div>
              )}
              {getActivePlayer().board.map(star => (
                 <Card key={star.id} card={star} location="board" onHover={setHoveredCard} onDrop={(e) => handleDropOnStar(e, star)} />
              ))}
           </div>
        </div>

        {/* Sidebar Right: Info */}
        <div className="w-80 bg-gray-900 border-l border-gray-800 flex flex-col z-20 shadow-2xl relative flex-none h-full min-h-0">
           <div className="h-2/5 p-4 border-b border-gray-800 bg-gray-800/50 flex-none overflow-hidden">
             {hoveredCard ? (
                <div className="h-full flex flex-col">
                   <div className="text-xl font-bold text-yellow-400 mb-1">{hoveredCard.name}</div>
                   <div className="text-xs text-gray-400 mb-4">{hoveredCard.type} {hoveredCard.classValue ? `• Class ${hoveredCard.classValue}` : ''}</div>
                   <div className="text-sm text-gray-200 leading-relaxed pr-2 mb-4">{hoveredCard.effectText}</div>
                   
                   {/* Attachments List */}
                   {(hoveredCard as BoardCard).attachments?.length > 0 && (
                       <div className="flex-1 overflow-y-auto border-t border-gray-700 pt-2">
                           <div className="text-[10px] font-bold text-gray-500 uppercase mb-2">Orbiting Bodies</div>
                           {(hoveredCard as BoardCard).attachments.map(a => (
                               <div key={a.id} className="flex items-center text-xs text-green-400 mb-1">
                                   <span className="mr-2">●</span> {a.name}
                               </div>
                           ))}
                       </div>
                   )}
                </div>
             ) : (
                <div className="h-full flex items-center justify-center text-gray-500 text-sm italic">Hover over a card</div>
             )}
           </div>
           
           <div className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-2 bg-black/40 min-h-0">
              {game.logs.map((log, idx) => <div key={idx} className="text-gray-400 border-l-2 border-gray-700 pl-2 py-1">{log}</div>)}
              <div ref={logsEndRef} />
           </div>

           <div className="p-4 border-t border-gray-800 bg-gray-900 flex-none">
              <div className="text-xs text-center text-gray-500 mb-2">
                  Systems Completed: <span className="text-yellow-500 font-bold">{getActivePlayer().systemsCompleted}/3</span>
              </div>
              <button 
                onClick={nextPhaseOrTurn}
                disabled={game.isResolving || !!game.pendingDecision}
                className="w-full py-3 bg-yellow-600 hover:bg-yellow-500 text-black font-bold rounded shadow-lg transition"
              >
                {game.phase === Phase.Action ? "End Turn" : "Pass Priority"}
              </button>
           </div>
        </div>
      </div>

      {/* Hand */}
      <div className="h-48 bg-black/80 backdrop-blur-md border-t border-gray-800 relative z-30 flex-none">
        <div className="absolute top-0 left-0 bg-yellow-600 text-black text-xs font-bold px-3 py-1 rounded-br-lg z-40">
           {getActivePlayer().name} (Active)
        </div>
        <div className="w-full h-full flex items-center justify-center overflow-x-auto pt-4 pb-2 px-10 gap-2">
           {getActivePlayer().hand.map(card => (
              <Card 
                key={card.id} 
                card={card} 
                location="hand" 
                onHover={setHoveredCard}
                onDragStart={handleDragStart}
                isDragging={draggedCard?.id === card.id}
                onClick={() => { if (game.selectionMode === 'DISCARD_TO_FIVE') handleDiscard(card); }}
              />
           ))}
        </div>
      </div>
    </div>
  );
}