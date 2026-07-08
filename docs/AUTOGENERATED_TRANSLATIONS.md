# Auto-generated language packs — review needed

Swedish (`sv`) is native-authored and verified. The packs below (`de`, `fi`, `no`,
`da`, `is`) were **auto-generated** and are marked `reviewed = false`. They run, but
their scores are provisional until a **native speaker** has checked them. At runtime,
using an unreviewed language emits a `UserWarning`.

Confidence, honestly: German / Norwegian / Danish are close and likely fine; **Finnish
and Icelandic need careful review** (Finnish is grammatically distant; `langdetect`
doesn't even support Icelandic, so it couldn't be auto-verified).

## Checks run

**Forward (langdetect on each pack's directive):** `da→da`, `de→de`, `fi→fi`, `no→no`,
`sv→sv` ✓. `is→no` (langdetect has no Icelandic; not a text error). No language was
detected as English.

**Structure:** every pack has all four metric sections; all metrics build prompts for all
six languages without error (unit-tested).

**Backward (gloss of the shared `keep_language` directive)** — compare the meaning to the
Swedish reference *"Everything you extract and write must be in Swedish, the same language
as the answer. Never translate into English."*

| Lang | Directive | English gloss (back-translation) |
| --- | --- | --- |
| `de` | Alles, was du extrahierst, und deine gesamte Ausgabe müssen auf Deutsch sein, in derselben Sprache wie die Antwort. Übersetze niemals ins Englische. | "Everything you extract and your entire output must be in German, in the same language as the answer. Never translate into English." |
| `no` | Alt du trekker ut og alt du skriver, skal være på norsk, samme språk som svaret. Oversett aldri til engelsk. | "Everything you extract and everything you write shall be in Norwegian, the same language as the answer. Never translate into English." |
| `da` | Alt hvad du uddrager, og alt hvad du skriver, skal være på dansk, samme sprog som svaret. Oversæt aldrig til engelsk. | "Everything you extract, and everything you write, shall be in Danish, the same language as the answer. Never translate into English." |
| `fi` | Kaiken, mitä poimit ja kirjoitat, on oltava suomeksi, samalla kielellä kuin vastaus. Älä koskaan käännä englanniksi. | "Everything you extract and write must be in Finnish, the same language as the answer. Never translate into English." |
| `is` | Allt sem þú dregur út og allt sem þú skrifar á að vera á íslensku, á sama tungumáli og svarið. Þýddu aldrei yfir á ensku. | "Everything you extract and write shall be in Icelandic, in the same language as the answer. Never translate into English." |

The glosses match the Swedish intent. The per-metric instructions in each
`prompts/<code>.toml` mirror the (reviewed) Swedish structure 1:1 — review those against
`prompts/sv.toml`.

## Review checklist (per language)

1. Grammar and natural phrasing of every string in `prompts/<code>.toml`.
2. The label terms (e.g. the word chosen for "reference answer" / *FACIT*) read naturally.
3. The `keep_language` directive is unambiguous.
4. When satisfied, set `reviewed = true` in that pack's TOML.
