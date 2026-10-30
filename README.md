# chess-ai 🏁

A complete chess engine in **pure Python, stdlib only** — no chess library, no dependencies.

## Correctness: perft-validated

Move generation is validated against the standard perft suite, including the notoriously
tricky [Kiwipete](https://www.chessprogramming.org/Perft_Results#Position_2) position
(castling rights, en-passant pins, promotions):

```
perft(1) =      20  expected      20  [PASS]
perft(2) =     400  expected     400  [PASS]
perft(3) =    8902  expected    8902  [PASS]
perft(4) =  197281  expected  197281  [PASS]
perft(3) Kiwipete = 97862  expected 97862  [PASS]
```

Full rules implemented: castling, en passant, promotion, check/checkmate/stalemate,
fifty-move rule, threefold repetition.

## Engine

- Iterative-deepening **negamax** with **alpha-beta pruning**
- **Quiescence search** on captures (no horizon effect on captures)
- **MVV-LVA** move ordering
- Evaluation: material + piece-square tables

## Usage

```bash
python chess_ai.py perft        # run correctness tests
python chess_ai.py selfplay 3   # engine vs engine at depth 3
python chess_ai.py play 3       # play against it (moves like e2e4)
```

See [demo_game.txt](demo_game.txt) for a full self-play game with scores and timing.

## License

MIT
