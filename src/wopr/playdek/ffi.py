"""ctypes binding for Playdek's `TwilightLib.dll`, the native engine behind the
Steam edition of Twilight Struggle.

The Unity app is only a front end: rules, card database (Lua, statically
linked) and the AI all live in this one DLL, exposed as a flat C API. The
signatures and struct layouts below were recovered from the app's IL2CPP
metadata (the C# `[DllImport("TwilightLib")]` declarations and the structs
they marshal); nothing here is documented by Playdek, so every layout is
cross-checked empirically by `wopr.playdek.smoke`.

Nothing from the game is redistributed: the DLL and its Lua database are
loaded from the user's own Steam install, located by `find_install()`.
"""

from __future__ import annotations

import ctypes as C
import os
from enum import IntEnum
from pathlib import Path

# --------------------------------------------------------------------------
# Locating the install
# --------------------------------------------------------------------------

INSTALL_ENV = "STRUGGLER_PLAYDEK_DIR"
_CANDIDATES = (
    r"C:\Program Files (x86)\Steam\steamapps\common\Twilight Struggle",
    r"C:\Program Files\Steam\steamapps\common\Twilight Struggle",
)
_DLL_RELATIVE = Path("TwilightStruggle_Data") / "Plugins" / "x86_64" / "TwilightLib.dll"
_LUA_RELATIVE = Path("TwilightStruggle_Data") / "StreamingAssets" / "Lua"


def find_install() -> Path:
    """The game's install directory: `$STRUGGLER_PLAYDEK_DIR`, else a scan of
    the usual Steam library roots on every drive letter."""
    env = os.environ.get(INSTALL_ENV)
    if env:
        root = Path(env)
        if not (root / _DLL_RELATIVE).is_file():
            raise FileNotFoundError(f"{INSTALL_ENV}={env}: no {_DLL_RELATIVE} there")
        return root
    candidates = list(_CANDIDATES)
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        candidates.append(rf"{letter}:\SteamLibrary\steamapps\common\Twilight Struggle")
    for cand in candidates:
        if (Path(cand) / _DLL_RELATIVE).is_file():
            return Path(cand)
    raise FileNotFoundError(
        "Twilight Struggle (Playdek) install not found; set " + INSTALL_ENV
    )


# --------------------------------------------------------------------------
# Enums (values from the IL2CPP metadata)
# --------------------------------------------------------------------------


class EPlayer(IntEnum):
    NONE = -1
    USSR = 0
    US = 1


class ECardLocation(IntEnum):
    NONE = -1
    USSRHAND = 0
    USHAND = 1
    DECK = 2
    DISCARDED = 3
    REMOVED = 4
    INPLAY = 5
    USSRHEADLINE = 6
    USAHEADLINE = 7


class EScenario(IntEnum):
    STANDARD = 0
    CCW = 1  # Chinese Civil War
    LATEWAR = 2
    TURNZERO = 3


class EChooseSidesMethod(IntEnum):
    CREATORUSSR = 0
    CREATORUS = 1
    RANDOM = 2
    BID = 3


class GameOverType(IntEnum):
    NONE = 0
    VICTORY_POINTS = 1
    EUROPE_CONTROL = 2
    FINAL_SCORING = 3
    DEFCON = 4
    HELD_CARDS = 5
    CUBAN_MISSILE_CRISIS = 6
    WARGAMES = 7
    FORFEIT = 8


class EventType(IntEnum):
    """`GameEvent.EventType`: the record kinds `UpdateGame` emits."""

    OUTPUT_PAUSE = 1
    OUTPUT_ANIMATION_CARD = 2
    OUTPUT_ANIMATION_ADD_INFLUENCE = 3
    OUTPUT_ANIMATION_REMOVE_INFLUENCE = 4
    OUTPUT_ANIMATION_TARGET_COUNTRY = 5
    OUTPUT_ANIMATION_VICTORY_POINTS = 6
    COUNTRY_DEFINITION = 7
    COUNTRY_INFLUENCE = 8
    CARD_LOCATION = 9
    CARD_IN_PLAY_STATUS = 10
    ACTION_ROUND = 11
    PHASING_PLAYER = 12
    TURN_NUMBER = 13
    VP_TRACK = 14
    DEFCON_LEVEL = 15
    MILITARY_OPS = 16
    SPACE_RACE_TRACK = 17
    CARDS_SPACED = 18
    CHINA_CARD = 19
    GAME_OVER = 20
    ASSIGN_SIDES = 21
    DISCARDS_RESHUFFLED = 22
    CARDS_ADDED = 23
    REGION_SCORING = 24
    CARD_PLAYED = 25
    EVENT_PLAYED = 26
    COUP = 27
    REALIGNMENT = 28
    PUSH_RESOLVE_CARD = 29
    POP_RESOLVE_CARD = 30
    PUSH_REVEAL_CARD = 31
    POP_REVEAL_CARD = 32
    SET_REVEAL_CARD_PLAYER = 33
    SET_HEADLINE_CARD_REVEALED = 34
    LOAD_PROGRESS = 35
    COMMIT_PLAYER_DECISION = 36
    COUP_ROLL = 37
    WAR_ROLL = 38
    SPACE_RACE_ROLL = 39
    TRAP_ROLL = 40
    SCORING_CARD_PLAYED = 41
    FINAL_SCORING = 42
    EFFECT_ROLL = 43
    END_TURN = 44
    HEADLINE_ANNOUNCE = 45
    RESHUFFLE = 46
    PAUSE_FOR_REVEALED_CARDS = 47
    TUTORIAL_AI_SELECTED_OPTION = 48
    BIDDING_RESULTS = 49
    TURN_ZERO = 50
    TURN_ZERO_CRISIS_CARD = 51
    SET_STATECRAFT_CARD_REVEALED = 52
    CRISIS_CARD_ROLL = 53
    ACHIEVEMENT = 54
    LOG_UPDATED = 99
    LOG_BEGIN_TURN = 100
    LOG_END_TURN = 101
    LOG_BEGIN_ACTION_ROUND = 102
    LOG_END_ACTION_ROUND = 103
    LOG_BEGIN_CARD_EVENT = 104
    LOG_END_CARD_EVENT = 105
    LOG_BEGIN_OPS = 106
    LOG_END_OPS = 107
    LOG_INFLUENCE_CHANGE = 108
    LOG_REVEAL_HEADLINE = 109
    LOG_COUP_RESULT = 110
    LOG_REALIGNMENT_RESULT = 111
    LOG_WAR_RESULT = 112
    LOG_MILITARY_OPS = 113
    LOG_DEFCON_LEVEL = 114
    LOG_VP_TRACK = 115
    LOG_SPACE_RACE_RESULT = 116
    LOG_SPACE_RACE_ADVANCE = 117
    LOG_DISCARD = 118
    LOG_CARD_IN_PLAY_STATUS = 119
    LOG_CANCEL_ACTION = 120
    LOG_REPORT_SIDES = 121
    LOG_REVEAL_CARD = 122
    LOG_PLAY_ADDITIONAL_CARD = 123
    LOG_TRAP_RESULT = 124
    LOG_BIDDING_RESULT = 125
    LOG_GRAIN_SALES_RESULT = 126
    LOG_CHERNOBYL = 127
    LOG_OLYMPICS = 128
    LOG_RESHUFFLE = 129


# --------------------------------------------------------------------------
# Structs (native layouts; managed offsets from the metadata, pointers where
# the C# side had arrays/strings)
# --------------------------------------------------------------------------


class GameParameters(C.Structure):
    _fields_ = [
        ("additionalCardFlags", C.c_uint32),
        ("scenario", C.c_int32),
        ("chooseSidesMethod", C.c_int32),
        ("additionalInfluence", C.c_int32),
    ]


class AppPlayerData(C.Structure):
    """One seat, as the app's IL2CPP wrapper marshals it: 52 bytes, the
    `ushort[]` fields by value (two entries each) and the name inline."""

    _pack_ = 1
    _fields_ = [
        ("id", C.c_int32),
        ("userAvatars", C.c_uint16 * 2),
        ("userRatings", C.c_uint16 * 2),
        ("playerType", C.c_int8),
        ("aiDifficultyLevel", C.c_int8),
        ("networkPlayerState", C.c_uint16),
        ("networkPlayerTimer", C.c_uint32),
        ("name", C.c_char * 32),
    ]


assert C.sizeof(AppPlayerData) == 52


class PlayerType(IntEnum):
    """`AppPlayerData.playerType`. `StartGame` routes the decisions of the
    `LOCAL` seat to the options listener; a `HOTSEAT` seat is the second
    human of a hotseat game (also routed there, in hotseat mode only)."""

    LOCAL = 0
    HOTSEAT = 1
    AI = 2


class AIDifficulty(IntEnum):
    """`AppPlayerData.aiDifficultyLevel` for `PlayerType.AI` (1 is unused and crashes)."""

    EASY = 0
    HARD = 2


class GamePlayerInfo(C.Structure):
    _fields_ = [
        ("userID", C.c_uint32),
        ("forfeit", C.c_uint32),
        ("avatarIndex0", C.c_uint16),
        ("avatarIndex1", C.c_uint16),
        ("displayName", C.c_char_p),
    ]


class GameOption(C.Structure):
    """One entry of the list handed to the options listener. 64 bytes on the
    wire, the label inline (the C# side marshals it to a `string`)."""

    _pack_ = 1
    _fields_ = [
        ("optionIndex", C.c_int32),
        ("selectionID", C.c_uint16),  # card / country id, per the decision kind
        ("selectionHint", C.c_uint16),  # the kind of thing being selected
        ("isHidden", C.c_uint8),
        ("optionText", C.c_char * 55),
    ]


assert C.sizeof(GameOption) == 64


class GamePlayerHandState(C.Structure):
    _fields_ = [
        ("userID", C.c_uint32),
        ("handCardCount", C.c_uint16),
        ("chinaCardFaceUp", C.c_uint16),
        ("chinaCardFaceDown", C.c_uint16),
    ]


class GamePlayerAIState(C.Structure):
    _fields_ = [
        ("userID", C.c_uint32),
        ("isAIPlayer", C.c_uint16),
        ("isAIThinking", C.c_uint16),
        ("aiThinkingPercentage", C.c_float),
    ]


class GameDeckCounts(C.Structure):
    _fields_ = [
        ("draw_pile_count", C.c_int32),
        ("discard_pile_count", C.c_int32),
        ("removed_pile_count", C.c_int32),
    ]


class GameTurnLogEntry(C.Structure):
    _fields_ = [
        ("logType", C.c_int32),
        ("logSourceInstanceID", C.c_int32),
        ("logTargetInstanceID", C.c_int32),
        ("logData", C.c_int32),
    ]


# Event payloads (namespace `GameEvent` in the metadata). All-int structs;
# the field order is the wire order. This covers every type the DLL emits:
# the app's `HandleEvent` has a case for 1-21 and 28-54 (22-27 fall to its
# default branch), 99 is the one log type that is an event, and 100+ are
# the `logType`s of `GetGameTurnLogBuffer` entries.
_PAYLOADS: dict[EventType, list[str]] = {
    EventType.OUTPUT_PAUSE: ["pause_type", "animation_data", "exclude_player_index"],
    EventType.OUTPUT_ANIMATION_CARD: [
        "card_instance_id", "animation_source_location", "animation_source_instance_id",
        "animation_destination_location", "animation_destination_instance_id", "animation_event_hint",
    ],
    EventType.OUTPUT_ANIMATION_ADD_INFLUENCE: [
        "source_player_index", "source_card_instance_id", "country_instance_id", "influence_count", "animation_event_hint",
    ],
    EventType.OUTPUT_ANIMATION_REMOVE_INFLUENCE: [
        "source_player_index", "source_card_instance_id", "country_instance_id", "influence_count", "animation_event_hint",
    ],
    EventType.OUTPUT_ANIMATION_TARGET_COUNTRY: [
        "source_player_index", "source_card_instance_id", "country_instance_id", "target_type", "animation_event_hint",
    ],
    EventType.OUTPUT_ANIMATION_VICTORY_POINTS: ["source_card_instance_id", "victory_point_count", "animation_event_hint"],
    EventType.COUNTRY_DEFINITION: ["id", "stability", "battleground"],
    EventType.COUNTRY_INFLUENCE: ["id", "ussr_influence", "us_influence"],
    EventType.CARD_LOCATION: ["id", "location", "bDoNotAnimate"],
    EventType.CARD_IN_PLAY_STATUS: ["cardinplay_instance_id", "sourcecard_instance_id", "owner_index", "duration_type", "inplay"],
    EventType.ACTION_ROUND: [
        "isSimulating", "action_round", "end_of_turn", "phasing_player_superpower", "player_ID",
        "affected_by_missile_envy", "affected_by_kremlin_flu", "scoring_card_count",
    ],
    EventType.PHASING_PLAYER: ["phasing_player"],
    EventType.TURN_NUMBER: ["isSimulating", "turn_number", "ussr_hand", "usa_hand", "has_extra_round"],
    EventType.VP_TRACK: ["vp_track"],
    EventType.DEFCON_LEVEL: ["isSimulating", "defcon_level"],
    EventType.MILITARY_OPS: ["ussr_milops", "us_milops"],
    EventType.SPACE_RACE_TRACK: ["ussr_space", "us_space"],
    EventType.CARDS_SPACED: ["ussr_cards_spaced", "us_cards_spaced"],
    EventType.CHINA_CARD: ["instanceID", "player", "faceup"],
    EventType.GAME_OVER: ["winner", "win_type"],
    EventType.ASSIGN_SIDES: ["USSRPlayerID"],
    EventType.DISCARDS_RESHUFFLED: ["zero"],
    EventType.SCORING_CARD_PLAYED: ["card_id", "player_id"],
    EventType.FINAL_SCORING: ["victory_points_ussr", "victory_points_usa", "european_control_winner"],
    EventType.REALIGNMENT: ["realign_player_index", "country", "USSR_roll_result", "US_roll_result"],
    EventType.PUSH_RESOLVE_CARD: ["card"],
    EventType.POP_RESOLVE_CARD: ["card"],
    EventType.PUSH_REVEAL_CARD: ["card", "player"],
    EventType.POP_REVEAL_CARD: ["card"],
    EventType.SET_REVEAL_CARD_PLAYER: ["player_index"],
    EventType.SET_HEADLINE_CARD_REVEALED: ["player_index", "revealed"],
    EventType.SET_STATECRAFT_CARD_REVEALED: ["player_index", "revealed"],
    EventType.LOAD_PROGRESS: ["progress"],  # float on the wire
    EventType.COMMIT_PLAYER_DECISION: ["moveCount", "winnerPlayerIndex"],
    EventType.COUP_ROLL: ["coup_player_index", "country_id", "roll"],
    EventType.WAR_ROLL: ["player_index", "country_id", "roll"],
    EventType.SPACE_RACE_ROLL: [
        "isSimulating", "roll", "success", "card", "space_race_player_index", "space_race_current_level",
        "space_race_opponent_level", "space_race_remaining_attempts", "space_race_next_level",
        "space_race_required_ops", "space_race_required_roll", "space_race_advance_victory_points",
        "space_race_advance_gain_bonus", "space_race_advance_remove_bonus",
    ],
    EventType.TRAP_ROLL: [
        "player_id", "roll", "trap_source_card_id", "trap_discard_card_id",
        "trap_required_operations_points", "trap_escape_roll_target",
    ],
    EventType.EFFECT_ROLL: ["card_id", "ussr_roll", "ussr_modify", "usa_roll", "usa_modify"],
    EventType.END_TURN: ["ussr_ops", "usa_ops", "defcon", "space_race"],
    EventType.HEADLINE_ANNOUNCE: ["space_race"],
    EventType.PAUSE_FOR_REVEALED_CARDS: ["player_index"],
    EventType.TUTORIAL_AI_SELECTED_OPTION: ["selection"],  # two u16 packed in one int
    EventType.BIDDING_RESULTS: [
        "player1_id", "player1_bid", "player1_bidSide", "player2_id", "player2_bid", "player2_bidSide", "tie_breaker",
    ],
    EventType.TURN_ZERO: ["begin"],
    EventType.TURN_ZERO_CRISIS_CARD: ["crisis_card_instance_id"],
    EventType.CRISIS_CARD_ROLL: [
        "crisis_result_index", "crisis_card_instance_id", "die_roll", "modifier_ussr", "modifier_usa", "final_result",
    ],
    EventType.ACHIEVEMENT: ["achievementID", "data"],
    EventType.LOG_UPDATED: ["update_count"],
    # No struct in the metadata; one int on the wire in every game seen.
    EventType.RESHUFFLE: ["data"],
}


class SelectionHint(IntEnum):
    """`GameOption.selectionHint`: what kind of thing an option selects.
    Collected from live games (`wopr.playdek.smoke` tallies them); the
    values read as `0xA000 | (kind << 4) | variant`."""

    STOP = 0xA000  # "No More Realignment": end an optional repetition early
    CANCEL = 0xA001
    PLAY_CARD = 0xA010  # "Play <card>" in "Play Your Action Round"
    PLAY_SCORING_CARD = 0xA011  # the same for a scoring card (also "Headline <scoring card>")
    SWITCH_CARD = 0xA013  # "Play <card>" offered again inside "Select Use For Event Card"
    HEADLINE_CARD = 0xA020
    PLAY_EVENT = 0xA021
    RESOLVE_EVENT_FIRST = 0xA022  # the opponent's event: before or after the Ops
    OPS_INFLUENCE = 0xA030  # "Place Influence" as the use of the Ops
    INFLUENCE_COUNTRY = 0xA031  # "Place Influence in <country>" with Ops
    SETUP_INFLUENCE_COUNTRY = 0xA032  # "Place Influence in <country>" for setup / an event
    REMOVE_INFLUENCE_COUNTRY = 0xA033
    RELOCATE_FROM_COUNTRY = 0xA034  # De-Stalinization: "Relocate Influence from <country>"
    OPS_REALIGNMENT = 0xA040
    OPS_COUP = 0xA050
    WAR_COUNTRY = 0xA052  # "War in <country>" (Indo-Pakistani War)
    OPS_SPACE_RACE = 0xA060
    DISCARD_CARD = 0xA091  # "Discard <card>" (Blockade's alternative)
    FORCED_DISCARD_CARD = 0xA09A  # "Discard <card>" when a discard is required (Bear Trap / Quagmire)
    EVENT_CHOICE = 0xA0A0  # an event's either/or ("Choose for Eastern Europe:")
    EVENT_CHOICE_YESNO = 0xA0F1  # "Participate in Olympic Games?" -> Participate / Boycott
    EVENT_CHOICE_HIDDEN = 0xA0FF  # the blank hidden entry that accompanies it


def payload_fields(kind: int) -> list[str] | None:
    try:
        return _PAYLOADS.get(EventType(kind))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Callback types
# --------------------------------------------------------------------------

# void (*)(int playerID, const char* prompt, int numOptions, GameOption* options)
GameOptionsListener = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_int, C.POINTER(GameOption))
# void (*)(void* saveData, int size, void* shortSave)
SaveWorldDataFunc = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_void_p)
# void (*)(const char* message) -- not declared on the C# side. Never seen
# called: the DLL prints its diagnostics (the Lua database load, Lua errors)
# straight to stdout instead.
DebugFunc = C.CFUNCTYPE(None, C.c_char_p)

EVENT_BUFFER_BYTES = 16384  # the app's `m_EventBufferLength`
STATE_BUFFER_BYTES = 512  # the app's `k_maxDataSize`


def load(install: Path | None = None) -> tuple[C.WinDLL, Path]:
    """Load the DLL and declare every prototype this package uses.

    Returns the library and the install root (the Lua database lives under
    it; see `TwilightLib.initialize`)."""
    root = install or find_install()
    lib = C.WinDLL(str(root / _DLL_RELATIVE))
    proto = {
        "Initialize": (None, [C.c_char_p, C.c_int]),
        "Shutdown": (None, []),
        "SetGameOptionsListener": (None, [GameOptionsListener]),
        "SetSaveDataFunc": (None, [SaveWorldDataFunc]),
        "SetDebugFunction": (None, [DebugFunc]),
        "SelectGameOption": (None, [C.c_int]),
        "SelectGameOptionWithData": (None, [C.c_int, C.c_uint]),
        "ResendGameOptionsList": (None, []),
        "StartGame": (None, [C.POINTER(GameParameters), C.c_int, C.POINTER(AppPlayerData), C.c_uint]),
        "NetworkCreate": (None, []),
        "NetworkDestroy": (None, []),
        "ExitCurrentGame": (None, []),
        "UpdateGame": (C.c_int, [C.c_void_p, C.c_int]),
        "ForceUpdateStateMachineInput": (C.c_int, [C.c_void_p, C.c_int]),
        "HasTemporaryMoveBuffer": (C.c_int, []),
        "CommitTemporaryMoveBuffer": (None, []),
        "RevertTemporaryMoveBuffer": (None, []),
        "GetGameParameters": (C.POINTER(GameParameters), []),
        "GetGamePlayerInfo": (C.c_int, [C.c_int, C.c_void_p, C.c_int]),
        "GetGamePlayerHandState": (C.c_int, [C.c_int, C.c_void_p, C.c_int]),
        "GetGamePlayerAIState": (C.c_int, [C.c_int, C.c_void_p, C.c_int]),
        "GetGameCurrentScore": (C.c_int, []),
        "GetPendingDefconLevel": (C.c_int, [C.c_int]),
        "GetGameDeckCounts": (C.c_int, [C.c_void_p, C.c_int]),
        "GetGameTurnLogCount": (C.c_int, []),
        "GetGameTurnLogBuffer": (C.c_int, [C.c_int, C.c_void_p, C.c_int]),
        "GetCardsInPlay": (C.c_int, [C.c_void_p, C.c_int]),
        "GetInstanceList": (C.c_int, [C.c_int, C.c_int, C.c_void_p, C.c_int]),
        "GetInstanceData": (C.c_int, [C.c_int, C.c_int, C.c_void_p, C.c_int]),
        "GetLocalPlayerIndex": (C.c_int, []),
        "GetNewLocalPlayerID": (C.c_int, []),
        "GetCurrentGameID": (C.c_uint, []),
    }
    for name, (restype, argtypes) in proto.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib, root


def lua_dir(root: Path) -> Path:
    return root / _LUA_RELATIVE
