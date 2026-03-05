"""
AVA (Atomic Visual Actions) Dataset Class Mappings
80 action classes for video action detection

Reference: https://research.google.com/ava/
"""

# Complete AVA v2.2 Action Class Labels
AVA_CLASSES = {
    1: "bend/bow (at the waist)",
    2: "crouch/kneel",
    3: "dance",
    4: "fall down",
    5: "flee",
    6: "jump/leap",
    7: "lie down",
    8: "martial art",
    9: "run/jog",
    10: "sit",
    11: "stand",
    12: "swim",
    13: "walk",
    14: "answer phone",
    15: "call on the phone",
    16: "carry/hold (an object)",
    17: "catch (an object)",
    18: "cut with knife",
    19: "jump down",
    20: "drink",
    21: "eat",
    22: "enter",
    23: "exit",
    24: "hit (an object)",
    25: "hit (a person)",
    26: "hold hands",
    27: "hug (a person)",
    28: "kick (an object)",
    29: "kick (a person)",
    30: "kiss",
    31: "lift (an object)",
    32: "look at a person",
    33: "play an instrument",
    34: "point to (an object)",
    35: "point to (a person)",
    36: "push (an object)",
    37: "push (a person)",
    38: "put down",
    39: "read",
    40: "ride",
    41: "sing",
    42: "sit down",
    43: "stand up",
    44: "take a photo",
    45: "throw",
    46: "touch (an object)",
    47: "touch (a person)",
    48: "turn around",
    49: "use a computer",
    50: "watch screen",
    51: "write",
    52: "texting",
    53: "fight (with person)",
    54: "hand shake",
    55: "high five",
    56: "nod",
    57: "shake head",
    58: "snap fingers",
    59: "wave",
    60: "cook",
    61: "cut vegetables",
    62: "flip pancake",
    63: "fry",
    64: "grill",
    65: "mix",
    66: "peel",
    67: "pour",
    68: "season",
    69: "slice",
    70: "smell",
    71: "taste",
    72: "throat cut",
    73: "tie something",
    74: "zip something",
    75: "carry something",
    76: "open something",
    77: "close something",
    78: "wash dishes",
    79: "wash hands",
    80: "watch TV"
}

# Reverse mapping (name to id)
CLASS_TO_ID = {v.lower(): k for k, v in AVA_CLASSES.items()}

# Short names for common actions (user-friendly)
COMMON_CLASSES = {
    "kiss": 30,
    "hug": 27,
    "walk": 13,
    "run": 9,
    "run/jog": 9,
    "sit": 10,
    "stand": 11,
    "jump": 6,
    "jump/leap": 6,
    "fight": 53,
    "dance": 3,
    "eat": 21,
    "drink": 20,
    "stand up": 43,
    "sit down": 42,
    "lie down": 7,
    "kick": 29,
    "punch": 25,
    "throw": 45,
    "catch": 17,
    "wave": 59,
    "hand shake": 54,
    "high five": 55,
    "nod": 56,
    "shake head": 57,
    "answer phone": 14,
    "call on the phone": 15,
    "read": 39,
    "write": 51,
    "texting": 52,
    "watch TV": 80,
    "watch screen": 50,
    "use a computer": 49,
    "cook": 60,
    "swim": 12,
    "ride": 40,
    "crouch": 2,
    "bend": 1,
    "bend/bow": 1,
    "fall down": 4,
    "flee": 5,
    "martial art": 8,
    "point to": 34,
    "hug person": 27,
    "hit": 25,
    "hit person": 25,
    "hit object": 24,
    "carry": 16,
    "lift": 31,
    "put down": 38,
    "enter": 22,
    "exit": 23,
    "turn around": 48,
    "look at person": 32,
    "take photo": 44,
    "play instrument": 33,
    "sing": 41
}

# Classes selected by user (kiss, hug, walk, run, sit, jump, fight, etc.)
USER_SELECTED_CLASSES = [
    "kiss", "hug", "walk", "run", "sit", "jump", "fight",
    "dance", "eat", "drink", "stand", "lie down", "kick",
    "throw", "catch", "wave", "hand shake", "high five",
    "run/jog", "stand up", "sit down", "punch", "bend",
    "crouch", "fall down", "flee", "martial art", "swim",
    "ride", "answer phone", "call on the phone", "read",
    "write", "texting", "watch TV", "use a computer"
]

# Get class IDs for user selected classes
def get_user_class_ids():
    """Get list of class IDs for user selected classes"""
    class_ids = []
    for class_name in USER_SELECTED_CLASSES:
        if class_name in COMMON_CLASSES:
            class_ids.append(COMMON_CLASSES[class_name])
        elif class_name in CLASS_TO_ID:
            class_ids.append(CLASS_TO_ID[class_name])
    return sorted(set(class_ids))

def get_class_ids_from_names(class_names):
    """Get class IDs from a list of class names"""
    class_ids = []
    for class_name in class_names:
        class_name_lower = class_name.lower().strip()
        if class_name_lower in COMMON_CLASSES:
            class_ids.append(COMMON_CLASSES[class_name_lower])
        elif class_name_lower in CLASS_TO_ID:
            class_ids.append(CLASS_TO_ID[class_name_lower])
    return sorted(set(class_ids))

# AVA dataset file URLs
AVA_ANNOTATIONS_URL = "https://storage.googleapis.com/ava-dataset/annotations"

# Required annotation files
ANNOTATION_FILES = {
    "train": "ava_train_v2.2.csv",
    "val": "ava_val_v2.2.csv",
    "test": "ava_test_v2.2.csv",
    "labels": "ava_action_list_v2.2.csv"
}

if __name__ == "__main__":
    print("AVA Dataset Classes:")
    print("=" * 50)
    print(f"Total classes: {len(AVA_CLASSES)}")
    print("\nUser selected classes:")
    user_ids = get_user_class_ids()
    print(user_ids)
    print("\nClass mapping:")
    for class_id in user_ids:
        print(f"  {class_id}: {AVA_CLASSES[class_id]}")
