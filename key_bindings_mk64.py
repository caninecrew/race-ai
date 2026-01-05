"""
Default keyboard bindings for mupen64plus SDL input (Mario Kart 64).

These map to the keycodes Mupen64Plus expects in InputAutoCfg.ini:
  Stick: arrow keys (Up/Down/Left/Right)
  A (accelerate): Z                -> keycode 122
  Z Trig (use item): Left Shift    -> keycode 304  (or pick another key)
  B (brake/reverse): X             -> keycode 120
  R (hop/power slide): S           -> keycode 115
  L (map toggle): A                -> keycode 97
  Start (pause): Enter/Return      -> keycode 13
  C-Buttons: I (Up/105), K (Down/107), J (Left/106), L (Right/108)
  D-Pad: disabled by default unless configured.

Update your ~/.config/mupen64plus/InputAutoCfg.ini [Keyboard] section with these keycodes
if your defaults don’t work.
"""

# Friendly mapping
BINDINGS = {
    "Stick": {
        "Up": "ArrowUp",
        "Down": "ArrowDown",
        "Left": "ArrowLeft",
        "Right": "ArrowRight",
    },
    "Buttons": {
        "A (accelerate)": "Z",
        "B (brake/reverse)": "X",
        "Z Trig (use item)": "LeftShift",
        "R (hop/power slide)": "S",
        "L (map toggle)": "A",
        "Start (pause)": "Enter/Return",
    },
    "C-Buttons": {
        "C-Up": "I",
        "C-Down": "K",
        "C-Left": "J",
        "C-Right": "L",
    },
}

# Keycodes to use in InputAutoCfg.ini
KEYCODES = {
    "Start": 13,
    "A Button (accelerate)": 122,  # Z
    "B Button (brake/reverse)": 120,  # X
    "Z Trig (use item)": 304,  # Left Shift
    "A Button (hop/slide)": 115,  # S
    "R Trig (map toggle)": 97,  # A
    "C Button U": 105,  # I
    "C Button D": 107,  # K
    "C Button L": 106,  # J
    "C Button R": 108,  # L
    "X Axis": (276, 275),  # Left, Right arrows
    "Y Axis": (273, 274),  # Up, Down arrows
    # D-Pad entries can be added if needed, defaults to disabled (0).
}


def pretty_print() -> None:
    print("Mario Kart 64 key bindings (SDL input):")
    for section, mapping in BINDINGS.items():
        print(f"{section}:")
        for action, key in mapping.items():
            print(f"  {action}: {key}")
    print("\nInputAutoCfg.ini keycodes:")
    for action, code in KEYCODES.items():
        print(f"  {action}: {code}")


if __name__ == "__main__":
    pretty_print()
