"""Hand-written opponent fields for stress-testing agents.

Two fields:

* COOPERATIVE_FIELD - 15 rule-based strategies of the kind most often proposed
  for the noisy three-player game: mostly forgiving retaliators, plus a few
  grim triggers and one mostly-defecting agent. Names describe the rule.
* HOSTILE_FIELD - 10 exploitative or unforgiving strategies, used to check
  what happens when most opponents are exploitative or unforgiving.

Every strategy is f(own_history, opp1_history, opp2_history) -> 0 (C) or 1 (D)
and sees only executed (post-noise) actions, exactly like a real agent.
"""
import random


class Scripted:
    def __init__(self, name, rule):
        self.name = name
        self.rule = rule

    def agentName(self):
        return self.name

    def play(self, mine, opp1, opp2):
        return self.rule(mine, opp1, opp2)


def _unprompted(o, others, t):
    """o defected at round t although nobody defected at round t-1."""
    return o[t] == 1 and (t == 0 or all(h[t - 1] == 0 for h in others))


# ---- cooperative field -------------------------------------------------

def grim_2_of_3(m, a, b):
    """Cooperate; switch to permanent defection once an opponent defects 2 of 3 rounds."""
    for o in (a, b):
        for t in range(2, len(o)):
            if sum(o[t - 2:t + 1]) >= 2:
                return 1
    return 0


def escalating_random(m, a, b):
    """Punishment probability grows with an opponent's defections; grim after 5."""
    d = max(sum(a), sum(b))
    if d >= 5:
        return 1
    if len(a) and (a[-1] or b[-1]):
        return int(random.random() < min(1, (d - 1) * 0.25))
    return 0


def tit_for_two_tats(m, a, b):
    """Defect only after an opponent defects twice in a row."""
    return int(any(len(o) >= 2 and o[-1] == o[-2] == 1 for o in (a, b)))


def window_punisher(m, a, b):
    """Defect if an opponent defected 2+ of the last 5, until both cooperate."""
    if len(a) and a[-1] == 0 and b[-1] == 0:
        return 0
    return int(any(sum(o[-5:]) >= 2 for o in (a, b)))


def unprompted_counter(m, a, b):
    """Track unprompted defections; punish repeat offenders and double defections."""
    t = len(m)
    for o, x in ((a, b), (b, a)):
        u = sum(_unprompted(o, [m, x], i) for i in range(t))
        if u > 3 and sum(o[-3:]) >= 1:
            return 1
        if len(o) >= 2 and o[-1] == o[-2] == 1:
            return 1
    return 0


def prober(m, a, b):
    """Probe after 20 cooperative rounds; 3-round punishment on 2-in-5."""
    t = len(m)
    for o in (a, b):
        for k in range(1, 4):
            if t - k >= 5 and sum(o[t - k - 5:t - k]) >= 2:
                return 1
    if t >= 20 and all(sum(r) == 0 for r in zip(m[-20:], a[-20:], b[-20:])) and t % 25 == 0:
        return 1
    return 0


def forgiving_rate_tracker(m, a, b):
    """Forgive singles; answer repeats or joint defection; defect vs D-heavy opponents."""
    t = len(m)
    if t >= 10 and max(sum(a), sum(b)) / t > 0.5:
        return 1
    if t and a[-1] == b[-1] == 1:
        return 1
    return int(any(len(o) >= 3 and sum(o[-3:]) >= 2 for o in (a, b)))


def tft_then_grim(m, a, b):
    """Tit for two tats, switching to grim after 4 defections by one opponent."""
    if max(sum(a), sum(b)) >= 4:
        return 1
    return tit_for_two_tats(m, a, b)


def three_round_punisher(m, a, b):
    """3 rounds of punishment after an opponent's 2nd defection within 10 rounds."""
    t = len(m)
    for k in range(0, 3):
        for o in (a, b):
            if t - k >= 1 and o[t - k - 1] == 1 and sum(o[max(0, t - k - 10):t - k]) >= 2:
                return 1
    return 0


def unprompted_only(m, a, b):
    """Punish only an unprompted defection, for one round."""
    t = len(m)
    for o, x in ((a, b), (b, a)):
        if t and o[-1] == 1 and _unprompted(o, [m, x], t - 1):
            return 1
    return 0


def tit_for_tat(m, a, b):
    """Defect if either opponent defected last round."""
    return int(len(a) > 0 and (a[-1] or b[-1]))


def grim_trigger(m, a, b):
    """Defect forever after any opponent defection (noise included)."""
    return int(1 in a or 1 in b)


def grim_2_in_a_row(m, a, b):
    """Defect forever once an opponent defects twice in a row."""
    return int(any(o[t] == o[t + 1] == 1 for o in (a, b) for t in range(len(o) - 1)))


def three_in_a_row(m, a, b):
    """Retaliate only after an opponent defects 3 rounds in a row."""
    return int(any(len(o) >= 3 and sum(o[-3:]) == 3 for o in (a, b)))


def conditional_defector(m, a, b):
    """Mostly defect; cooperate only once an opponent cooperates >70% of the time."""
    t = len(m)
    if t < 5:
        return 1
    return int(not (max((t - sum(a)) / t, (t - sum(b)) / t) > 0.7 and random.random() > 0.05))


COOPERATIVE_FIELD = [Scripted(n, f) for n, f in [
    ("Grim after 2-of-3", grim_2_of_3),
    ("Escalating random retaliator", escalating_random),
    ("Tit for two tats", tit_for_two_tats),
    ("Window punisher", window_punisher),
    ("Unprompted-defection counter", unprompted_counter),
    ("Prober", prober),
    ("Forgiving rate tracker", forgiving_rate_tracker),
    ("TF2T then grim", tft_then_grim),
    ("3-round punisher", three_round_punisher),
    ("Unprompted-only punisher", unprompted_only),
    ("Tit for Tat", tit_for_tat),
    ("Grim trigger", grim_trigger),
    ("Grim after 2-in-a-row", grim_2_in_a_row),
    ("Retaliate after 3-in-a-row", three_in_a_row),
    ("Conditional defector", conditional_defector),
]]


# ---- hostile field --------------------------------------------------------

def always_defect(m, a, b):
    return 1


def random_play(m, a, b):
    return random.randint(0, 1)


def sneaky(m, a, b):
    """Defect once every 7 rounds, spaced out to look like noise."""
    return int(len(m) % 7 == 3)


def probe_and_exploit(m, a, b):
    """Defect twice early; keep defecting 30% of the time if nobody retaliated."""
    return int(len(m) in (5, 6) or (len(m) > 6 and (a[5] or a[6]) == 0 and random.random() < 0.3))


def endgame_defector(m, a, b):
    """Cooperate, then defect every round from round 85."""
    return int(len(m) >= 85)


def always_cooperate(m, a, b):
    return 0


HOSTILE_FIELD = [Scripted(n, f) for n, f in [
    ("Always Defect", always_defect),
    ("Random", random_play),
    ("Sneaky 1-in-7", sneaky),
    ("Probe and exploit", probe_and_exploit),
    ("Endgame defector", endgame_defector),
    ("Conditional defector", conditional_defector),
    ("Tit for Tat", tit_for_tat),
    ("Grim trigger", grim_trigger),
    ("Tit for two tats", tit_for_two_tats),
    ("Always Cooperate", always_cooperate),
]]
