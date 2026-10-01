# Decision Lab static site

`docs/` is the GitHub Pages document root. It uses plain HTML, CSS and JavaScript modules and reads published evidence from `docs/data/`. No server or browser model connection is required.

Serve from the repository root for local preview:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000/docs/`. A `file://` preview cannot fetch the JSON evidence.

The browser 2048 and Connect Four engines are human-play sandboxes. They never populate measured model results. Connect Four uses deterministic depth-4 minimax with center-first ties. 2048 uses a seeded xorshift generator for repeatable browser restarts; that generator is separate from recorded benchmark spawning.

Run the rule checks with:

```sh
node --test docs/assets/game-engines.test.mjs
```

The static UI expects `tetris-summary.json`, `tetris-replays.json`, `arcade-summary.json`, `arcade-replays.json` and `blockstar-source.json` under `data/`. Missing or empty evidence is explicitly unmeasured. The site does not fabricate model decisions or substitute the human-play baseline.
