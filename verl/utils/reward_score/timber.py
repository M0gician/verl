import re
import difflib

BACKTRACK_CUES = [
    "This doesn't seem right. I am restarting from the last correct step and think again:",
    "Wait, let me try again:",
    "Alternatively...",
    "Feel like I'm missing something.",
    "Hmm...",
    "Something is off, let me try again."
]


def compute_score(solution_str: str, ground_truth: str) -> float:
    score = 0.0
    try:
        score += acc_score(solution_str, ground_truth)
        score += early_stopping_reward(solution_str, ground_truth)
        score += backtrack_repetition_loss(solution_str)
    except Exception as e:
        return 0.0
    
    return score


def solution_parsing_nl(solution_str: str, backtrack_cues: list[str]) -> list[str]:
    """
    Splits a given solution string by the provided backtrack_cues (case insensitive).
    """
    if not solution_str:
        return []
    if not backtrack_cues:
        return [solution_str]

    # Create a regex pattern that ORs all cues, case insensitive
    pattern = "|".join(re.escape(cue) for cue in backtrack_cues)
    # Split the string using the pattern, re.IGNORECASE for case insensitivity
    parts = re.split(f"({pattern})", solution_str, flags=re.IGNORECASE)

    result = []
    current_segment = ""
    for i, part in enumerate(parts):
        is_delimiter = any(cue.lower() == part.lower() for cue in backtrack_cues)
        if i % 2 == 1 and is_delimiter:
            if current_segment.strip():
                result.append(current_segment.strip())
            current_segment = ""
        else:
            current_segment += part

    if current_segment.strip():
        result.append(current_segment.strip())
    
    # Handle cases where the string might start or end with a cue, or have consecutive cues
    # The re.split approach with filtering should handle this, but let's ensure no empty strings.
    return [segment for segment in result if segment]


def backtrack_repetition_loss(solution_str: str, similarity_threshold: float = 0.7) -> float:
    segments = solution_parsing_nl(solution_str, BACKTRACK_CUES)
    
    if not segments or len(segments) < 2:
        return 0.0  # No repetitions possible if less than 2 segments

    total_penalty_score = 0.0
    
    # Normalize all segments first to avoid repeated calls to strip_string
    normalized_segments = [strip_string(seg) for seg in segments]

    # Iterate from the second segment onwards
    for i in range(1, len(normalized_segments)):
        current_segment_normalized = normalized_segments[i]
        if not current_segment_normalized:  # Skip if current segment is empty after normalization
            continue

        # Compare with all preceding segments
        for j in range(i):
            preceding_segment_normalized = normalized_segments[j]
            if not preceding_segment_normalized:  # Skip if preceding segment is empty
                continue

            similarity = difflib.SequenceMatcher(None, current_segment_normalized, preceding_segment_normalized).ratio()

            if similarity > similarity_threshold:
                total_penalty_score -= 0.3  # Apply fixed penalty for the current segment
                break  # Penalty for current_segment_normalized applied, move to the next i
                
    final_score = max(-1.0, total_penalty_score)
    return final_score


def early_stopping_reward(solution_str: str, ground_truth: str) -> float:
    raw_boxed_expressions = _extract_all_boxed_expressions(solution_str)
    if not raw_boxed_expressions:
        return 0.0

    innermost_answers = [_get_innermost_content(expr) for expr in raw_boxed_expressions]
    
    # Filter out any potential empty strings after getting innermost content, though unlikely
    innermost_answers = [ans for ans in innermost_answers if ans]

    if not innermost_answers:
        return 0.0

    num_answers = len(innermost_answers)
    first_match_idx = -1

    for i, answer_content in enumerate(innermost_answers):
        if is_equiv(answer_content, ground_truth):
            first_match_idx = i
            break
    
    if first_match_idx == -1: # No match found
        return 0.0
    
    # Check if the first match is the latest answer
    if first_match_idx == num_answers - 1:
        return 1.0
    else:
        # First match is not the latest, check the next answer
        if first_match_idx + 1 < num_answers:
            next_answer_content = innermost_answers[first_match_idx + 1]
            if is_equiv(next_answer_content, ground_truth):
                return 0.1
        # If no next answer or next answer doesn't match
        return 0.0


def _extract_all_boxed_expressions(solution_str: str) -> list[str]:
    r"""
    Extracts all \boxed{...}, \fbox{...}, and \boxed ... expressions from a string, in order.
    Handles nested braces for \boxed{...} and \fbox{...}.
    For \boxed ..., it captures until the next $ or end of string.
    """
    # Regex to find \boxed{...}, \fbox{...}, or \boxed ...
    # For \boxed{...} and \fbox{...}:
    #   - \\boxed{ or \\fbox{ : literal start
    #   - (       : start of capture group for content
    #   -   (?:   : non-capturing group for alternatives
    #   -     [^{}] : match any char that is not a brace
    #   -     |     : OR
    #   -     {     : match an opening brace
    #   -       [^{}]* : match any char that is not a brace (content of inner brace)
    #   -     }     : match a closing brace
    #   -   )*?   : repeat the non-capturing group zero or more times, non-greedily
    #   - )       : end of capture group for content
    #   - }       : literal closing brace
    # For \boxed ...:
    #   - \\boxed\s+ : literal \boxed followed by one or more spaces
    #   - (.*?)      : capture everything non-greedily (content)
    #   - (?=\s*\$|$) : positive lookahead for optional space then $ or end of string
    # The outer parentheses in the pattern make each full match a group.
    pattern = r"(\\boxed{(?:[^{}]|{[^{}]*})*})|(\\fbox{(?:[^{}]|{[^{}]*})*})|(\\boxed\s+.*?(?=\s*\$|$))"
    
    matches = re.finditer(pattern, solution_str)
    extracted_expressions = []
    for match in matches:
        # The finditer will return a match object.
        # We need to find which group actually matched to get the full expression.
        if match.group(1): # \boxed{...}
            extracted_expressions.append(match.group(1))
        elif match.group(2): # \fbox{...}
            extracted_expressions.append(match.group(2))
        elif match.group(3): # \boxed ...
            extracted_expressions.append(match.group(3).strip()) # Strip trailing spaces for \boxed ...
            
    return extracted_expressions


def _get_innermost_content(raw_expr_str: str) -> str:
    """
    Repeatedly applies remove_boxed to get the innermost content of a boxed expression.
    """
    content = raw_expr_str
    # Keep removing layers as long as it's a recognized boxed expression
    while True:
        try:
            # Attempt to remove a layer
            next_content = remove_boxed(content)
            if next_content == content: # No change means it's no longer a recognized boxed form
                break
            content = next_content
        except ValueError: # remove_boxed raises ValueError if not a recognized form
            break 
    return content


def acc_score(solution_str: str, ground_truth: str) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)
    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s: str) -> str:
    """
    Removes one layer of \boxed{}, \fbox{}, or \boxed (space) from a string.
    Expects `s` to be a full boxed expression.
    """
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    elif s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    elif s.startswith("\\fbox{") and s.endswith("}"):
        return s[len("\\fbox{"):-1]
    else:
        # This case should ideally not be hit if this function is fed
        # valid full boxed expressions from an extractor.
        raise ValueError(f"Input to remove_boxed is not a recognized boxed expression: {s}")


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace(r"\\%", "")
    string = string.replace(r"\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
