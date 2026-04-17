"""
Information-Theoretic Evaluation Metrics

Computes per-game metrics from episode logs (EpisodeLogger.to_dict() format):

1. Mutual Information I(H; A)
   How much a player's actions reveal about their actual hand.
   Lower = better bluffer (actions decouple from hand contents).

2. KL Divergence D_KL(P_true || P_believed)
   How far opponent beliefs shift from reality after seeing claims.
   Higher = more effective deception.

3. Belief Misalignment
   Average absolute gap between claimed count and actual matching cards.

4. TrueSkill Rating
   Bayesian multiplayer skill rating.
"""
from __future__ import annotations
import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1 & 2. Mutual Information + KL Divergence
# ---------------------------------------------------------------------------

class InformationTheoreticMetrics:

    def compute_mutual_information(self, turns: List[dict], player_id: int) -> float:
        """
        Approximate I(H; A) from episode log turns for one player.

        H = was the play honest (bool)
        A = n_cards claimed (observable action)

        Lower MI = actions don't leak honesty = better bluffer.
        """
        player_plays = [
            t for t in turns
            if t.get("player") == player_id
            and t.get("action", {}).get("type") == "play"
            and "honest" in t.get("event", {})
        ]
        if not player_plays:
            return 0.0

        joint: Dict[Tuple[bool, int], int] = defaultdict(int)
        for t in player_plays:
            ev = t["event"]
            honest = ev["honest"]
            n = min(ev.get("n_cards", 1), 4)
            joint[(honest, n)] += 1

        total = len(player_plays)
        joint_probs = {k: v / total for k, v in joint.items()}

        h_marginal: Dict[bool, float] = defaultdict(float)
        a_marginal: Dict[int, float] = defaultdict(float)
        for (h, a), p in joint_probs.items():
            h_marginal[h] += p
            a_marginal[a] += p

        def _entropy(dist: dict) -> float:
            return -sum(p * math.log2(p + 1e-10) for p in dist.values() if p > 0)

        h_a = _entropy(a_marginal)
        h_a_given_h = 0.0
        for h_val in (True, False):
            ph = h_marginal.get(h_val, 0.0)
            if ph == 0:
                continue
            cond: Dict[int, float] = defaultdict(float)
            for (h, a), p in joint_probs.items():
                if h == h_val:
                    cond[a] += p / ph
            h_a_given_h += ph * _entropy(cond)

        return max(0.0, h_a - h_a_given_h)

    def compute_kl_divergence(self, turns: List[dict], player_id: int) -> float:
        """
        D_KL(P_true || P_believed)

        P_true  = distribution of ranks actually played by this player
        P_believed = distribution of ranks claimed by this player

        Higher KL = their claims successfully misled opponents about their hand.
        """
        from engine.card import RANKS
        all_ranks = RANKS

        true_counts: Dict[str, int] = defaultdict(int)
        claimed_counts: Dict[str, int] = defaultdict(int)

        for t in turns:
            if t.get("player") != player_id:
                continue
            ev = t.get("event", {})
            if ev.get("action", {}).get("type") != "play" and t.get("action", {}).get("type") != "play":
                continue
            # Actual cards played (from event — only visible in log)
            for card_str in ev.get("actual_cards", []):
                rank = card_str[:-1] if len(card_str) > 1 else card_str  # strip suit
                # Handle multi-char ranks like "10"
                for r in RANKS:
                    if card_str.startswith(r):
                        rank = r
                        break
                true_counts[rank] += 1
            # Claimed rank
            claimed_rank = ev.get("claimed_rank") or t.get("action", {}).get("claimed_rank")
            if claimed_rank:
                n = ev.get("n_cards", 1)
                claimed_counts[claimed_rank] += n

        true_total = sum(true_counts.values()) or 1
        claimed_total = sum(claimed_counts.values()) or 1

        kl = 0.0
        for rank in all_ranks:
            p_true = true_counts.get(rank, 0) / true_total
            p_believed = max(claimed_counts.get(rank, 0) / claimed_total, 1e-6)
            if p_true > 0:
                kl += p_true * math.log2(p_true / p_believed)

        return max(0.0, kl)

    def compute_belief_misalignment(self, turns: List[dict], player_id: int) -> float:
        """Average |claimed_count - actual_matching_count| per play turn."""
        player_plays = [
            t for t in turns
            if t.get("player") == player_id
            and t.get("action", {}).get("type") == "play"
        ]
        if not player_plays:
            return 0.0

        total = 0.0
        for t in player_plays:
            ev = t.get("event", {})
            claimed_rank = ev.get("claimed_rank", "")
            claimed_n = ev.get("n_cards", 0)
            actual = ev.get("actual_cards", [])
            matching = sum(1 for cs in actual if cs.startswith(claimed_rank))
            total += abs(claimed_n - matching)

        return total / len(player_plays)

    def analyze_episode(self, log: dict, player_id: int) -> dict:
        """Full information-theoretic analysis of one episode for one player."""
        turns = log.get("turns", [])
        outcome = log.get("outcome", {})

        mi   = self.compute_mutual_information(turns, player_id)
        kl   = self.compute_kl_divergence(turns, player_id)
        bm   = self.compute_belief_misalignment(turns, player_id)

        player_plays = [
            t for t in turns
            if t.get("player") == player_id
            and t.get("action", {}).get("type") == "play"
        ]
        bluffs = sum(1 for t in player_plays if not t.get("event", {}).get("honest", True))
        bluff_rate = bluffs / max(len(player_plays), 1)

        # Challenges issued by this player and their outcomes
        my_challenges = [
            t for t in turns
            if t.get("player") == player_id
            and t.get("action", {}).get("type") == "challenge"
        ]
        challenges_won = sum(
            1 for t in my_challenges
            if t.get("event", {}).get("challenge_result") == "caught_bluffing"
        )
        chal_acc = challenges_won / max(len(my_challenges), 1)

        # Bluffs caught (this player was the claimer and got challenged successfully)
        bluffs_caught = sum(
            1 for t in turns
            if t.get("event", {}).get("action", {}).get("type") == "challenge"
            and t.get("event", {}).get("pile_goes_to") == player_id
            and t.get("event", {}).get("challenge_result") == "caught_bluffing"
        )
        bluff_catch_rate = bluffs_caught / max(bluffs, 1)

        return {
            "mutual_information":  mi,
            "kl_divergence":       kl,
            "belief_misalignment": bm,
            "bluff_rate":          bluff_rate,
            "bluff_catch_rate":    bluff_catch_rate,
            "challenge_accuracy":  chal_acc,
            "won": outcome.get("winner") == player_id,
        }


# ---------------------------------------------------------------------------
# 3. TrueSkill Rating
# ---------------------------------------------------------------------------

@dataclass
class TrueSkillPlayer:
    name: str
    mu: float = 25.0
    sigma: float = 25.0 / 3
    games_played: int = 0

    @property
    def rating(self) -> float:
        return self.mu - 3 * self.sigma

    @property
    def display(self) -> str:
        return f"μ={self.mu:.1f} σ={self.sigma:.1f} → {self.rating:.1f}"


class TrueSkillRating:
    BETA     = 25.0 / 6
    TAU      = 25.0 / 300
    DRAW_PROB = 0.0

    def __init__(self):
        self.players: Dict[str, TrueSkillPlayer] = {}

    def get_or_create(self, name: str) -> TrueSkillPlayer:
        if name not in self.players:
            self.players[name] = TrueSkillPlayer(name=name)
        return self.players[name]

    def update(self, ranked_names: List[str]) -> None:
        players = [self.get_or_create(n) for n in ranked_names]
        for p in players:
            p.sigma = math.sqrt(p.sigma ** 2 + self.TAU ** 2)
            p.games_played += 1
        winner = players[0]
        for loser in players[1:]:
            self._update_pair(winner, loser)

    def _update_pair(self, winner: TrueSkillPlayer, loser: TrueSkillPlayer) -> None:
        c  = math.sqrt(2 * self.BETA ** 2 + winner.sigma ** 2 + loser.sigma ** 2)
        t  = (winner.mu - loser.mu) / c
        e  = math.exp(-t ** 2 / 2) / (math.sqrt(2 * math.pi) * (0.5 * (1 + math.erf(t / math.sqrt(2)))))
        ws2, ls2 = winner.sigma ** 2, loser.sigma ** 2
        winner.mu  += (ws2 / c) * e
        loser.mu   -= (ls2 / c) * e
        winner.sigma = max(math.sqrt(ws2 * (1 - (ws2 / c ** 2) * e * (e + t))), 0.1)
        loser.sigma  = max(math.sqrt(ls2 * (1 - (ls2 / c ** 2) * e * (e + t))), 0.1)

    def leaderboard(self) -> List[TrueSkillPlayer]:
        return sorted(self.players.values(), key=lambda p: -p.rating)

    def print_leaderboard(self) -> None:
        print("\n=== TrueSkill Leaderboard ===")
        print(f"{'Rank':<5} {'Player':<22} {'Rating':>8} {'μ':>8} {'σ':>8} {'Games':>7}")
        print("-" * 60)
        for i, p in enumerate(self.leaderboard(), 1):
            print(f"{i:<5} {p.name:<22} {p.rating:>8.1f} {p.mu:>8.1f} {p.sigma:>8.2f} {p.games_played:>7}")
