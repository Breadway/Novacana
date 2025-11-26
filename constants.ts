import { CardDefinition, CardType } from './types';

// Raw JSON from requirements
export const RAW_CARD_LIBRARY: any[] = [
  // Star Cards (formerly Nexus)
  { name: "Proto-Star", type: "Star", quantity: 4, effectText: "Action: Draw 2 cards OR Search deck for 'White Dwarf' and add to hand." },
  { name: "Yellow Dwarf", type: "Star", classValue: 4, quantity: 3, effectText: "Class 4 Star." },
  { name: "Red Giant", type: "Star", classValue: 5, quantity: 3, effectText: "Class 5 Star." },
  { name: "Red Supergiant", type: "Star", classValue: 6, quantity: 2, effectText: "Class 6 Star." },
  { name: "White Dwarf", type: "Star", classValue: 3, quantity: 3, effectText: "Class 3 Star." },
  { name: "Red Dwarf", type: "Star", classValue: 2, quantity: 4, effectText: "Class 2 Star." },
  { name: "Brown Dwarf", type: "Star", classValue: 2, quantity: 5, effectText: "Class 2 Star." },

  // Discovery Cards (formerly Node)
  { name: "Asteroid Belt", type: "Discovery", quantity: 5, effectText: "Attach to Star." },
  { name: "Dwarf Planet", type: "Discovery", quantity: 5, effectText: "Attach to Star." },
  { name: "Rocky Planet", type: "Discovery", quantity: 5, effectText: "Attach to Star." },
  { name: "Gas Giant", type: "Discovery", quantity: 5, effectText: "Attach to Star." },
  { name: "Comet", type: "Discovery", quantity: 5, effectText: "Attach to Star." },

  // Flare Cards
  { name: "Supernova - Fusion", type: "Flare", subType: "Supernova", quantity: 2, effectText: "Win Condition Part 1." },
  { name: "Supernova - Gravity", type: "Flare", subType: "Supernova", quantity: 2, effectText: "Win Condition Part 2." },
  { name: "Supernova - Gas", type: "Flare", subType: "Supernova", quantity: 2, effectText: "Win Condition Part 3. Play all 3 unique Supernovas together to WIN." },
  { name: "Cosmic Denial", type: "Flare", quantity: 4, effectText: "Negate a Flare or Fracture. Destroy it." },
  { name: "Ignition", type: "Flare", quantity: 2, effectText: "Destroy opponent's Gas Giant. Search deck for Yellow Dwarf to hand." },
  { name: "Astral Projection", type: "Flare", quantity: 2, effectText: "Opponent reveals hand for 2 turns." },
  { name: "Stellar Alignment", type: "Flare", quantity: 5, effectText: "Draw 2 cards." },
  { name: "Wormhole", type: "Flare", quantity: 2, effectText: "Steal 1 random card from opponent hand." },

  // Omen Cards
  { name: "Solar Eclipse", type: "Omen", quantity: 4, effectText: "Attach to Completed System. It cannot be destroyed." },
  { name: "Planet of Life", type: "Omen", quantity: 2, effectText: "Attach to Rocky Planet. Once per turn: Reveal opponent hand." },

  // Fracture Cards
  { name: "Black Hole", type: "Fracture", quantity: 2, effectText: "Board Wipe: Reshuffle all Hands, Stars, and Discoveries into deck. Skip next 3 turns." },
  { name: "Athena", type: "Fracture", quantity: 4, effectText: "Destroy target Discovery OR Return Gas Giant to hand." },
  { name: "Pulsar", type: "Fracture", quantity: 2, effectText: "Reshuffle your Hand/Board into deck. Draw that many cards." },
  { name: "Dark Matter Void", type: "Fracture", quantity: 2, effectText: "Name a Class. Opponent cannot play Star of that class next turn." },
  { name: "Quantum Entanglement Failure", type: "Fracture", quantity: 2, effectText: "Both players select 1 random card from hand and swap." },
  { name: "Stellar Nursery Collapse", type: "Fracture", quantity: 2, effectText: "Mill top 5 cards of deck." }
];
