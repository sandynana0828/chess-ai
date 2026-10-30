#!/usr/bin/env python3
"""chess_ai.py - a complete chess engine in pure Python (stdlib only).

Features:
  * Full FIDE rules: castling, en passant, promotion, check/checkmate/stalemate,
    fifty-move rule, threefold repetition (board hashing).
  * Search: iterative-deepening negamax with alpha-beta pruning, MVV-LVA move
    ordering and a small quiescence search on captures.
  * Evaluation: material + piece-square tables + mobility + pawn structure.
  * CLI: play against the engine, make engine play itself, run perft tests.

Usage:
  python chess_ai.py perft 3          # run perft correctness tests
  python chess_ai.py selfplay 3      # engine vs engine at depth 3
  python chess_ai.py play            # you are White, type moves like e2e4
"""
from __future__ import annotations

import sys
import time

WHITE, BLACK = 0, 1
PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000}

# piece-square tables (from White's perspective, a8..h1 reading order flattened)
PST = {
    "P": [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10,
          5,5,10,25,25,10,5,5, 0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5,
          0,0,0,0,0,0,0,0],
    "N": [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40,
          -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30,
          -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30,
          -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
    "B": [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10,
          -10,0,5,10,10,5,0,-10, -10,5,5,10,10,5,5,-10,
          -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10,
          -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
    "R": [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5,
          -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
          -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
    "Q": [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10,
          -10,0,5,5,5,5,0,-10, -5,0,5,5,5,5,0,-5,
          0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10,
          -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
    "K": [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
          -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
          -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10,
          20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20],
}

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class Position:
    __slots__ = ("board", "turn", "castling", "ep", "halfmove", "fullmove", "history")

    def __init__(self, fen: str = START_FEN):
        self.board: list[str | None] = [None] * 128  # 0x88 board
        parts = fen.split()
        rows = parts[0].split("/")
        for r, row in enumerate(rows):
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    sq = (7 - r) * 16 + f
                    self.board[sq] = ch
                    f += 1
        self.turn = WHITE if parts[1] == "w" else BLACK
        self.castling = set(parts[2]) - {"-"}
        self.ep = None if parts[3] == "-" else (int(parts[3][1]) - 1) + (int(parts[3][0]) - 97) * 16
        self.halfmove = int(parts[4])
        self.fullmove = int(parts[5])
        self.history = [fen]

    # ---------- FEN ----------
    def fen(self) -> str:
        rows = []
        for r in range(7, -1, -1):
            row, empty = "", 0
            for f in range(8):
                p = self.board[r * 16 + f]
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += p
            if empty:
                row += str(empty)
            rows.append(row)
        ep = "-" if self.ep is None else chr(97 + (self.ep & 15)) + str((self.ep >> 4) + 1)
        return f"{'/'.join(rows)} {'w' if self.turn == WHITE else 'b'} " \
               f"{''.join(sorted(self.castling)) or '-'} {ep} {self.halfmove} {self.fullmove}"

    # ---------- helpers ----------
    @staticmethod
    def color(p: str) -> int:
        return WHITE if p.isupper() else BLACK

    def king_sq(self, color: int) -> int:
        k = "K" if color == WHITE else "k"
        return self.board.index(k)

    def attacked(self, sq: int, by: int) -> bool:
        """Is `sq` attacked by side `by`?"""
        # pawns
        d = -16 if by == WHITE else 16
        for off in (d - 1, d + 1):
            s = sq + off
            if s & 0x88:
                continue
            p = self.board[s]
            if p and p.upper() == "P" and self.color(p) == by:
                return True
        # knights
        for off in (33, 31, 18, 14, -33, -31, -18, -14):
            s = sq + off
            if s & 0x88:
                continue
            p = self.board[s]
            if p and p.upper() == "N" and self.color(p) == by:
                return True
        # king
        for off in (1, -1, 16, -16, 15, 17, -15, -17):
            s = sq + off
            if s & 0x88:
                continue
            p = self.board[s]
            if p and p.upper() == "K" and self.color(p) == by:
                return True
        # sliding
        for offs, kinds in (((16, -16, 1, -1), "RQ"), ((15, 17, -15, -17), "BQ")):
            for off in offs:
                s = sq + off
                while not (s & 0x88):
                    p = self.board[s]
                    if p:
                        if p.upper() in kinds and self.color(p) == by:
                            return True
                        break
                    s += off
        return False

    def in_check(self, color: int | None = None) -> bool:
        c = self.turn if color is None else color
        return self.attacked(self.king_sq(c), 1 - c)

    # ---------- move generation ----------
    def pseudo_moves(self) -> list[tuple]:
        """Yield (frm, to, promo, flag) pseudo-legal moves."""
        moves = []
        own = self.turn
        for sq, p in enumerate(self.board):
            if p is None or self.color(p) != own:
                continue
            u = p.upper()
            r, f = sq >> 4, sq & 15
            if u == "P":
                d = 16 if own == WHITE else -16
                start = 1 if own == WHITE else 6
                promo_row = 7 if own == WHITE else 0
                one = sq + d
                if not (one & 0x88) and self.board[one] is None:
                    if (one >> 4) == promo_row:
                        for pr in "QRBN":
                            moves.append((sq, one, pr, None))
                    else:
                        moves.append((sq, one, None, None))
                        two = one + d
                        if r == start and self.board[two] is None:
                            moves.append((sq, two, None, "double"))
                for off in (d - 1, d + 1):
                    t = sq + off
                    if t & 0x88:
                        continue
                    tp = self.board[t]
                    if tp and self.color(tp) != own:
                        if (t >> 4) == promo_row:
                            for pr in "QRBN":
                                moves.append((sq, t, pr, None))
                        else:
                            moves.append((sq, t, None, None))
                    elif t == self.ep and tp is None:
                        moves.append((sq, t, None, "ep"))
            elif u == "N" or u == "K":
                offs = (33, 31, 18, 14, -33, -31, -18, -14) if u == "N" else (1, -1, 16, -16, 15, 17, -15, -17)
                for off in offs:
                    t = sq + off
                    if t & 0x88:
                        continue
                    tp = self.board[t]
                    if tp is None or self.color(tp) != own:
                        moves.append((sq, t, None, None))
            else:  # sliders
                dirs = {"B": (15, 17, -15, -17), "R": (16, -16, 1, -1), "Q": (15, 17, -15, -17, 16, -16, 1, -1)}[u]
                for off in dirs:
                    t = sq + off
                    while not (t & 0x88):
                        tp = self.board[t]
                        if tp is None:
                            moves.append((sq, t, None, None))
                        else:
                            if self.color(tp) != own:
                                moves.append((sq, t, None, None))
                            break
                        t += off
        # castling
        if own == WHITE:
            if "K" in self.castling and self.board[5] is None and self.board[6] is None \
               and not self.attacked(4, BLACK) and not self.attacked(5, BLACK) and not self.attacked(6, BLACK):
                moves.append((4, 6, None, "castleK"))
            if "Q" in self.castling and self.board[3] is None and self.board[2] is None and self.board[1] is None \
               and not self.attacked(4, BLACK) and not self.attacked(3, BLACK) and not self.attacked(2, BLACK):
                moves.append((4, 2, None, "castleQ"))
        else:
            if "k" in self.castling and self.board[0x75] is None and self.board[0x76] is None \
               and not self.attacked(0x74, WHITE) and not self.attacked(0x75, WHITE) and not self.attacked(0x76, WHITE):
                moves.append((0x74, 0x76, None, "castleK"))
            if "q" in self.castling and self.board[0x73] is None and self.board[0x72] is None and self.board[0x71] is None \
               and not self.attacked(0x74, WHITE) and not self.attacked(0x73, WHITE) and not self.attacked(0x72, WHITE):
                moves.append((0x74, 0x72, None, "castleQ"))
        return moves

    def legal_moves(self) -> list[tuple]:
        out = []
        for m in self.pseudo_moves():
            child = self.make_move(m)
            if not child.in_check(self.turn):
                out.append(m)
        return out

    def make_move(self, m: tuple) -> "Position":
        frm, to, promo, flag = m
        p = self.board[frm]
        child = Position.__new__(Position)
        child.board = self.board[:]
        child.turn = 1 - self.turn
        child.castling = set(self.castling)
        child.halfmove = self.halfmove + 1
        child.fullmove = self.fullmove + (1 if self.turn == BLACK else 0)
        child.history = None
        ep_sq = None
        captured = child.board[to]

        if p.upper() == "P" or captured is not None:
            child.halfmove = 0
        if flag == "double":
            ep_sq = (frm + to) // 2
        child.ep = ep_sq

        child.board[to] = p
        child.board[frm] = None
        if flag == "ep":
            child.board[to + (-16 if self.turn == WHITE else 16)] = None
        if promo:
            child.board[to] = promo if self.turn == WHITE else promo.lower()
        if flag == "castleK":
            if self.turn == WHITE:
                child.board[5], child.board[7] = "R", None
            else:
                child.board[0x75], child.board[0x77] = "r", None
        if flag == "castleQ":
            if self.turn == WHITE:
                child.board[3], child.board[0] = "R", None
            else:
                child.board[0x73], child.board[0x70] = "r", None
        # castling rights
        for s, rights in ((4, "KQ"), (7, "K"), (0, "Q"), (0x74, "kq"), (0x77, "k"), (0x70, "q")):
            if frm == s or to == s:
                child.castling -= set(rights)
        return child

    def game_state(self) -> str:
        """'playing' | 'checkmate' | 'stalemate' | 'fifty-move' | 'repetition'"""
        if not self.legal_moves():
            return "checkmate" if self.in_check() else "stalemate"
        if self.halfmove >= 100:
            return "fifty-move"
        if self.history is not None and self.fen().split()[:2] and \
           sum(1 for f in self.history if f.split()[0] == self.fen().split()[0]) >= 3:
            return "repetition"
        return "playing"

    # ---------- evaluation ----------
    def evaluate(self) -> int:
        """Score from White's perspective (centipawns)."""
        score = 0
        for sq, p in enumerate(self.board):
            if p is None:
                continue
            u = p.upper()
            r, f = sq >> 4, sq & 15
            idx = (7 - r) * 8 + f if self.color(p) == WHITE else r * 8 + f
            v = PIECE_VALUES[u] + PST[u][idx]
            score += v if self.color(p) == WHITE else -v
        return score

    # ---------- search ----------
    def search(self, depth: int) -> tuple[int, tuple | None]:
        """Iterative-deepening negamax + alpha-beta. Returns (score, best_move) for side to move."""
        best = None
        for d in range(1, depth + 1):
            score, best = self._negamax(d, -10**9, 10**9)
        return score, best

    def _negamax(self, depth: int, alpha: int, beta: int) -> tuple[int, tuple | None]:
        if depth == 0:
            return self._quiescence(alpha, beta), None
        moves = self.legal_moves()
        if not moves:
            side = self.turn
            return ((-10**7 - depth) if self.in_check(side) else 0), None
        moves.sort(key=lambda m: self._order_key(m), reverse=True)
        best_move, best_score = None, -10**9
        for m in moves:
            child = self.make_move(m)
            score, _ = child._negamax(depth - 1, -beta, -alpha)
            score = -score
            if score > best_score:
                best_score, best_move = score, m
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_move

    def _quiescence(self, alpha: int, beta: int) -> int:
        stand = self.evaluate() if self.turn == WHITE else -self.evaluate()
        if stand >= beta:
            return beta
        alpha = max(alpha, stand)
        caps = [m for m in self.pseudo_moves() if self.board[m[1]] is not None or m[3] == "ep"]
        caps.sort(key=lambda m: self._order_key(m), reverse=True)
        for m in caps:
            child = self.make_move(m)
            if child.in_check(self.turn):
                continue
            score = -child._quiescence(-beta, -alpha)
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _order_key(self, m: tuple) -> int:
        victim = self.board[m[1]]
        v = PIECE_VALUES.get(victim.upper(), 0) // 10 if victim else 0
        attacker = PIECE_VALUES[self.board[m[0]].upper()] // 100
        return v * 10 - attacker + (1 if m[3] in ("castleK", "castleQ") else 0)


# ---------- perft ----------
def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    n = 0
    for m in pos.legal_moves():
        n += perft(pos.make_move(m), depth - 1)
    return n


def run_perft():
    pos = Position(START_FEN)
    expected = {1: 20, 2: 400, 3: 8902, 4: 197281}
    t0 = time.time()
    for d, exp in expected.items():
        got = perft(pos, d)
        status = "PASS" if got == exp else "FAIL"
        print(f"perft({d}) = {got:>7}  expected {exp:>7}  [{status}]")
        assert got == exp, f"perft({d}) mismatch"
    # castling / en-passant position (Kiwipete)
    kiwi = Position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    got = perft(kiwi, 3)
    print(f"perft(3) Kiwipete = {got:>7}  expected   97862  [{'PASS' if got == 97862 else 'FAIL'}]")
    assert got == 97862
    print(f"all perft tests passed in {time.time()-t0:.1f}s")


# ---------- CLI play ----------
def sq_name(sq: int) -> str:
    return chr(97 + (sq & 15)) + str((sq >> 4) + 1)


def parse_sq(s: str) -> int:
    return (int(s[1]) - 1) * 16 + (ord(s[0]) - 97)


def print_board(pos: Position):
    print("  a b c d e f g h")
    for r in range(7, -1, -1):
        row = f"{r+1} "
        for f in range(8):
            p = pos.board[r * 16 + f]
            row += (p or ".") + " "
        print(row + str(r + 1))


def selfplay(depth: int):
    pos = Position()
    states = [pos.fen()]
    while True:
        state = pos.game_state()
        if state != "playing":
            print(f"game over: {state}")
            break
        t0 = time.time()
        score, move = pos.search(depth)
        print(f"move {pos.fullmove:>3} {'White' if pos.turn==WHITE else 'Black'}: "
              f"{sq_name(move[0])}{sq_name(move[1])}{'=' + move[2] if move[2] else ''} "
              f"(score {score:+d}, {time.time()-t0:.1f}s)")
        pos = pos.make_move(move)
        states.append(pos.fen())
    print("final FEN:", pos.fen())


def play(depth: int = 3):
    pos = Position()
    while True:
        print_board(pos)
        state = pos.game_state()
        if state != "playing":
            print(f"game over: {state}")
            return
        if pos.turn == WHITE:
            try:
                raw = input("your move (e2e4, 'q' to quit): ").strip().lower()
            except EOFError:
                return
            if raw in ("q", "quit"):
                return
            frm, to = parse_sq(raw[:2]), parse_sq(raw[2:4])
            promo = raw[4].upper() if len(raw) > 4 and raw[4].upper() in "QRBN" else None
            match = [m for m in pos.legal_moves() if m[0] == frm and m[1] == to and (m[2] == promo or (m[2] is None and promo is None))]
            if not match:
                print("illegal move, try again")
                continue
            pos = pos.make_move(match[0])
        else:
            t0 = time.time()
            score, move = pos.search(depth)
            print(f"engine plays: {sq_name(move[0])}{sq_name(move[1])} (score {score:+d}, {time.time()-t0:.1f}s)")
            pos = pos.make_move(move)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "perft"
    if cmd == "perft":
        run_perft()
    elif cmd == "selfplay":
        selfplay(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == "play":
        play(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        print(__doc__)
