# Demo video: what to film and what to say (target 2:45, limit 3:00)

The story in one line: **a small AI on its own ships broken parts; inside fixed rules, with a
human in charge, it ships none, and it all runs offline on a laptop.**

## What you need
- The laptop on the **charger**, Power Mode = **Performance** (on battery Gemma times out)
- The USB webcam above the board, framed as before (screws on the pink crosses)
- Boards: one **good**, one with a **screw missing**, one with a **crack**
- A **phone** for 2 short clips of the real setup
- One person to speak. Record the voice separately if that's easier (a quiet room, phone voice memo is fine)

## Recording the screen
- Ubuntu: press **Print Screen** → click the **video camera icon** → choose **whole screen** →
  record. Stop with the red button in the top-right corner. Files go to `~/Videos/Screencasts`.
- Record each scene as its own clip. Re-take freely.
- Join clips, cut the waiting, and add text in a simple editor (CapCut, Kdenlive or Shotcut).
  When you cut a wait, put **"sped up"** or **"⏩ 2 min"** on screen. Never hide that it takes time.

## Set up before pressing record (every command from inside the project folder)
```
cd ~/"Kaggle MV Comp"
.venv/bin/streamlit run dashboard.py          # terminal 1, leave it running
```
```
cd ~/"Kaggle MV Comp"
bash demo/demo.sh clear                       # terminal 2: empty dashboard for a clean start
.venv/bin/python live.py                      # real Gemma
```
Place the dashboard (browser, http://localhost:8501) on the right half of the screen and the camera
window on the left half. Type your name in the dashboard sidebar.

Turn **Wi-Fi off** now (top-right menu) and leave it off for the whole recording.

---

## Scene 1: the problem (0:00–0:20)
**Show:** the notebook (section 6) table "naive Gemma said" vs "our cell", or `results/real/comparison.png`.

> "We asked Gemma 3, a small AI model, to inspect this board. It said *'all screws in place, 95%
> sure'*, but a screw was missing. Trusted alone, it let 12 of 16 defects through. So we kept
> Gemma, but put it inside a system of fixed rules and a human operator, so its mistakes get caught."

## Scene 2: the setup (0:20–0:35)
**Show:** phone clip: the webcam above the board, the laptop beside it. Then the screen: the
top-right corner with **Wi-Fi off**.

> "A webcam looks down at a board with four screws. The cell decides: pass, reject, or a person
> checks it. Everything runs on this laptop's CPU, no cloud, no internet; Wi-Fi is off."

## Scene 3: missing screw, instant (0:35–1:00)
**Show:** place the board with a **missing screw**, press **SPACE**. REJECT appears at once; the
dashboard shows the red box on the empty hole.

> "Fixed rules come first. Is the photo sharp enough? Where is each screw? Is it really there?
> That's measurement, not AI, so a missing screw is rejected in a tenth of a second and it shows
> exactly where."

## Scene 4: crack, the AI's job (1:00–1:35)
**Show:** place the **cracked** board, press SPACE. Film "INSPECTING… 5s", then **cut** ("⏩ sped up")
to the REJECT. In the dashboard point at the red tile and the **AI description** column.

> "Is this a crack? That's a judgement call, so it goes to Gemma, one small close-up at a time. It
> must describe what it sees before it answers, and a rule checks the answer format. Anything
> it can't answer cleanly goes to a person."

## Scene 5: good board and the human (1:35–2:10)
**Show:** place the **good** board, SPACE, cut the wait.
- If it **PASSES**: show all-green boxes.
  > "A board passes only when every area is checked and the fixed check and the AI agree on every screw."
- If it says **REVIEW** (an unsure screw) or **REJECT** (a cardboard fold): even better, use it.
  > "It isn't sure, so it doesn't guess. A person decides." Click **Override: PASS** in the
  > dashboard and show the History tab: *override by (your name)*. "Every decision is logged with who made it."

## Scene 6: e-stop (2:10–2:20)
**Show:** click **E-STOP** in the dashboard (red banner). Press SPACE in the camera window:
`LINE STOPPED: operator e-stop engaged`. Click **Clear e-stop**.

> "And the operator can stop the line at any moment; it halts before anything moves."

## Scene 7: results and close (2:20–2:45)
**Show:** `results/real/comparison.png`, then the Kaggle notebook page.

> "On 26 real photos it had never seen: 85% correct and zero defects shipped. The naive AI shipped
> 75% of them. Our mistakes are over-cautious, and a person fixes them in seconds. A small AI is
> useful inside fixed rules with a human in charge. Everything is in our public notebook."

---

## Optional extra line (if you have time; good for "industrial impact")
While showing the camera window's top bar:
> "Moving to a new camera, the screw check is re-calibrated from about ten photos in three
> minutes; the framing guide tells the operator how to place the camera."

## Numbers you may say (held-out real test set, 26 photos)
- Ours: **85%** correct, **0 of 16** defects shipped, 3 of 8 good boards rejected (cardboard folds), 12% sent to a person
- Naive Gemma: 46% correct, **12 of 16** defects shipped
- Trained classifier baseline: 77% correct, 1 of 16 shipped, 4 of 8 good boards rejected
- Missing screw: 0.1 s; full check of a good board: about 2.5 min on a laptop CPU

Don't claim numbers from the live demo itself; it's a demonstration, not the test.

## If something goes wrong while filming
- Gemma times out (REVIEW "AI unavailable") → is the charger in? Re-take.
- Screws show orange/red on a good board → glare: shade the lamp, or press G a few times and run
  `.venv/bin/python live.py --calibrate`.
- Live demo won't behave → `bash demo/demo.sh restore` loads 5 pre-recorded real results into the
  dashboard and you can film scenes 3–5 from the History tab instead.

## Upload
YouTube → visibility **Public** (required). Title: "Trusted Inspection Cell: offline part inspection
with Gemma 3 and a behaviour tree". Paste the link into the Kaggle Writeup.
