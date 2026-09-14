""""""
from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd
WORD = re.compile("[a-z0-9']+")
STOP = set("the a an and or of to in is are was were be been am being on for with as at by this that these those it its do does did so we you i he she they them his her him your our their what how when which who why not no if then than from up out about can will would could should i'm you're it's that's there here have has had get got me my we'll let's go going just like one two also into over under more most".split())
FILLERS = {'um', 'uh', 'er', 'erm', 'hmm', 'mm', 'mmm', 'oh', 'ah', 'yeah', 'okay', 'ok'}
ACK = {'yeah', 'yep', 'yes', 'ok', 'okay', 'mhm', 'mm', 'mmhmm', 'uh-huh', 'uhhuh', 'right', 'sure', 'cool', 'alright', 'yup', 'k', 'fine', 'true', 'exactly'}
CONFUSION = re.compile("\\b(i don't get|don't understand|confus|stuck|i'm lost|i am lost|no idea|i don't know|idk|what do you mean|don't know how|not sure how|can you repeat)\\b")
RESOLUTION = re.compile('\\b(i see|makes sense|got it|i get it|understand now|now i get|now i understand|oh i see|that makes sense|i understand)\\b')
SELF_EXPL = re.compile("\\b(because|therefore|since|that means|which means|thus|that's why|the reason|so that|cos|cus|coz|which gives|which leaves|giving us)\\b")
HEDGE = re.compile("\\b(maybe|i think|i guess|not sure|probably|perhaps|might be|i'm not sure)\\b")
ELICIT = re.compile("\\b(why|how|what do you think|can you explain|explain|what's next|walk me through|show me how|show your working|show your work|your turn|any thoughts|tell me why|tell me how|justify|prove|what makes)\\b")
TELL = re.compile("\\b(the answer is|answer is|you just|you simply|all you have|all you need|you have to|you need to|it's just|so it's just|just multiply|just add|just divide|first you|then you|step 1|step one)\\b")
CONFIRM = re.compile("\\b(exactly|correct|well done|good job|perfect|brilliant|that's right|that is right|spot on|good|great|nice|fantastic)\\b")
CORRECTIVE = re.compile("\\b(not quite|try again|not right|that's wrong|that is wrong|incorrect|not exactly|that's not|almost)\\b")
MATH = re.compile('(\\d|plus|minus|times|divid|multipl|equals?|subtract|\\badd\\b|sum|product|fraction|percent|decimal|remainder|squared?)')
WH_DEEP = re.compile('\\b(why|how)\\b')
WH_SHALLOW = re.compile("\\b(what('s| is) the answer|is it|what is it|is that right)\\b|(right|correct)\\s*\\?\\s*$")
GIVES_ANSWER = re.compile("\\b(the answer is|answer is|the result is|the solution is|it equals|it'?s? equal to|that equals|so it'?s?\\s+\\d|the value is|which gives us|that gives us)\\b", flags=re.IGNORECASE)
CFU = re.compile('(makes? sense|got it\\b|got that\\b|with me\\b|are you with|does that\\b|are you following|you following|\\bright\\?|\\bok\\?|\\bokay\\?|\\bclear\\?|understand\\?)')
DIALOGUE_COLS = ['stu_content_words', 'stu_ttr', 'substantive_turn_rate', 'ack_only_rate', 'longest_stu_utt_words', 'stu_words_final_third_share', 'stu_words_trend', 'confusion_rate', 'resolution_rate', 'ends_resolved', 'ends_confused', 'last_confusion_pos', 'recovered', 'final_third_stu_q_rate', 'stu_q_rate_delta', 'tutor_elicit_rate', 'tutor_tell_rate', 'elicit_to_tell', 'tutor_q_rate', 'uptake', 'revoicing_rate', 'confirm_rate', 'corrective_rate', 'wait_time_s', 'self_explanation_rate', 'math_doing_rate', 'hedge_rate', 'help_deep_rate', 'help_shallow_rate', 'gap_cv', 'pace_delta']
LO_COLS = ['lo_cov_session', 'lo_cov_student', 'lo_cov_end']

def parse_ts(s) -> float:
    if not isinstance(s, str):
        return math.nan
    p = s.strip().split(':')
    if len(p) != 3:
        return math.nan
    try:
        return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
    except ValueError:
        return math.nan

def _content_tokens(text: str) -> set:
    return {w for w in WORD.findall(text) if w not in STOP and w not in FILLERS}

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _slope(vals) -> float:
    """"""
    if len(vals) < 3:
        return 0.0
    x = np.linspace(0, 1, len(vals))
    return float(np.polyfit(x, np.asarray(vals, dtype=float), 1)[0])

def _empty() -> dict:
    d = {c: 0.0 for c in DIALOGUE_COLS}
    d['last_confusion_pos'] = -1.0
    d['_tok_all'] = set()
    d['_tok_stu'] = set()
    d['_tok_end'] = set()
    return d

def _session_arrays(tdf: pd.DataFrame):
    """"""
    if tdf.empty:
        return None
    roles = tdf['role'].fillna('').to_numpy()
    low = np.array([str(t).lower() for t in tdf['content'].fillna('').to_numpy()])
    secs = tdf['timestamp'].map(parse_ts).to_numpy()
    n = len(roles)
    third = n / 3.0
    pos = np.arange(n)
    seg = np.where(pos < third, 0, np.where(pos < 2 * third, 1, 2))
    words = np.array([WORD.findall(t) for t in low], dtype=object)
    wc = np.array([len(w) for w in words])
    toks = [set((w for w in ws if w not in STOP and w not in FILLERS)) for ws in words]
    si = np.where(roles == 'student')[0]
    ti = np.where(roles == 'tutor')[0]
    return {'roles': roles, 'low': low, 'secs': secs, 'n': n, 'seg': seg, 'words': words, 'wc': wc, 'toks': toks, 'si': si, 'ti': ti}

def dialogue_stats(tdf: pd.DataFrame) -> dict:
    """"""
    a = _session_arrays(tdf)
    if a is None:
        return _empty()
    roles, low, secs, n, seg = (a['roles'], a['low'], a['secs'], a['n'], a['seg'])
    words, wc, toks, si, ti = (a['words'], a['wc'], a['toks'], a['si'], a['ti'])
    d = {}
    all_tok = set().union(*toks) if toks else set()
    stu_tok = set().union(*[toks[i] for i in si]) if len(si) else set()
    end_tok = set().union(*[toks[i] for i in np.where(seg == 2)[0]]) or set()
    d['_tok_all'], d['_tok_stu'], d['_tok_end'] = (all_tok, stu_tok, end_tok)
    stu_wc = wc[si]
    stu_words = int(stu_wc.sum())
    stu_content = sum((len(toks[i]) for i in si))
    stu_tok_total = sum((len(toks[i]) for i in si))
    d['stu_content_words'] = float(stu_content)
    d['stu_ttr'] = len(stu_tok) / stu_tok_total if stu_tok_total else 0.0
    d['substantive_turn_rate'] = float(np.mean(stu_wc > 5)) if len(si) else 0.0
    ack = [bool(len(words[i]) <= 3 and words[i] and all((w in ACK for w in words[i]))) for i in si]
    d['ack_only_rate'] = float(np.mean(ack)) if ack else 0.0
    d['longest_stu_utt_words'] = float(stu_wc.max()) if len(si) else 0.0
    late_stu = stu_wc[seg[si] == 2].sum()
    d['stu_words_final_third_share'] = late_stu / stu_words if stu_words else 0.0
    d['stu_words_trend'] = _slope(stu_wc) if len(si) >= 3 else 0.0
    conf = np.array([bool(CONFUSION.search(low[i])) for i in si])
    reso = np.array([bool(RESOLUTION.search(low[i])) for i in si])
    d['confusion_rate'] = float(conf.mean()) if len(si) else 0.0
    d['resolution_rate'] = float(reso.mean()) if len(si) else 0.0
    last3 = si[-3:] if len(si) >= 3 else si
    d['ends_resolved'] = float(any((RESOLUTION.search(low[i]) for i in last3)))
    d['ends_confused'] = float(any((CONFUSION.search(low[i]) for i in last3)))
    conf_pos = [i for i in si if CONFUSION.search(low[i])]
    reso_pos = [i for i in si if RESOLUTION.search(low[i])]
    d['last_confusion_pos'] = conf_pos[-1] / n if conf_pos else -1.0
    d['recovered'] = float(bool(conf_pos) and bool(reso_pos) and (reso_pos[-1] > conf_pos[-1]))
    stu_q = np.array([low[i].rstrip().endswith('?') for i in si])
    late_mask = seg[si] == 2
    early_mask = seg[si] == 0
    d['final_third_stu_q_rate'] = float(stu_q[late_mask].mean()) if late_mask.any() else 0.0
    early_q = float(stu_q[early_mask].mean()) if early_mask.any() else 0.0
    d['stu_q_rate_delta'] = d['final_third_stu_q_rate'] - early_q
    if len(ti):
        tut_low = low[ti]
        elicit = np.array([bool(ELICIT.search(t)) for t in tut_low])
        tell = np.array([bool(TELL.search(t)) for t in tut_low])
        d['tutor_elicit_rate'] = float(elicit.mean())
        d['tutor_tell_rate'] = float(tell.mean())
        d['elicit_to_tell'] = float(elicit.sum()) / (tell.sum() + 1)
        d['tutor_q_rate'] = float(np.mean([t.rstrip().endswith('?') for t in tut_low]))
        d['confirm_rate'] = float(np.mean([bool(CONFIRM.search(t)) for t in tut_low]))
        d['corrective_rate'] = float(np.mean([bool(CORRECTIVE.search(t)) for t in tut_low]))
        ups = [_jaccard(toks[i], toks[i - 1]) for i in ti if i > 0 and roles[i - 1] == 'student']
        d['uptake'] = float(np.mean(ups)) if ups else 0.0
        d['revoicing_rate'] = float(np.mean([u > 0.3 for u in ups])) if ups else 0.0
        waits = []
        for i in ti:
            if low[i].rstrip().endswith('?') and i + 1 < n and (roles[i + 1] == 'student'):
                g = secs[i + 1] - secs[i]
                if np.isfinite(g) and g >= 0:
                    waits.append(g)
        d['wait_time_s'] = float(np.mean(waits)) if waits else 0.0
    else:
        for c in ['tutor_elicit_rate', 'tutor_tell_rate', 'elicit_to_tell', 'tutor_q_rate', 'confirm_rate', 'corrective_rate', 'uptake', 'revoicing_rate', 'wait_time_s']:
            d[c] = 0.0
    if len(si):
        d['self_explanation_rate'] = float(np.mean([bool(SELF_EXPL.search(low[i])) for i in si]))
        d['math_doing_rate'] = float(np.mean([bool(MATH.search(low[i])) for i in si]))
        d['hedge_rate'] = float(np.mean([bool(HEDGE.search(low[i])) for i in si]))
        q_idx = [i for i in si if low[i].rstrip().endswith('?')]
        if q_idx:
            d['help_deep_rate'] = float(np.mean([bool(WH_DEEP.search(low[i])) for i in q_idx]))
            d['help_shallow_rate'] = float(np.mean([bool(WH_SHALLOW.search(low[i])) for i in q_idx]))
        else:
            d['help_deep_rate'] = d['help_shallow_rate'] = 0.0
    else:
        for c in ['self_explanation_rate', 'math_doing_rate', 'hedge_rate', 'help_deep_rate', 'help_shallow_rate']:
            d[c] = 0.0
    valid = np.sort(secs[np.isfinite(secs)])
    gaps = np.diff(valid)
    gaps = gaps[gaps >= 0]
    d['gap_cv'] = float(gaps.std() / (gaps.mean() + 1e-06)) if len(gaps) else 0.0

    def _pace(segid):
        s = secs[(seg == segid) & np.isfinite(secs)]
        span = (s.max() - s.min()) / 60.0 if len(s) >= 2 else 0.0
        return np.sum(seg == segid) / span if span > 0 else 0.0
    d['pace_delta'] = _pace(2) - _pace(0)
    return d

def build_dialogue_matrix(feats: pd.DataFrame, store=None, show_progress: bool=False) -> pd.DataFrame:
    """"""
    from . import data as _data
    from . import features as _features
    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats['session_id']))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc='sessions')
        except ImportError:
            pass
    sess_rows, tok = ({}, {})
    for sid, tdf in it:
        try:
            d = dialogue_stats(tdf)
        except Exception:
            d = dialogue_stats(tdf.iloc[0:0])
        tok[sid] = (d.pop('_tok_all'), d.pop('_tok_stu'), d.pop('_tok_end'))
        sess_rows[sid] = d
    sess = pd.DataFrame.from_dict(sess_rows, orient='index').reindex(columns=DIALOGUE_COLS)
    lo_rows = [lo_alignment(r.learning_objective, *tok.get(r.session_id, (set(), set(), set()))) for r in feats.itertuples(index=False)]
    lo = pd.DataFrame(lo_rows, index=feats['response_id'].to_numpy(), columns=LO_COLS)
    meta = _features.build_meta_matrix(feats)
    out = meta.join(feats.set_index('response_id')['session_id']).join(sess, on='session_id').drop(columns='session_id').join(lo)
    out[DIALOGUE_COLS] = out[DIALOGUE_COLS].fillna(0.0)
    out[LO_COLS] = out[LO_COLS].fillna(0.0)
    return out[['learning_objective_id', 'learning_objective'] + DIALOGUE_COLS + LO_COLS]

def build_loc_matrix(feats: pd.DataFrame, idf: dict, idf_default: float, store=None, show_progress: bool=False) -> pd.DataFrame:
    """"""
    if idf is None:
        out = pd.DataFrame([_loc_empty() for _ in range(len(feats))], index=feats['response_id'].to_numpy())
        out.index.name = 'response_id'
        return out.reindex(columns=LOC_COLS)
    from . import data as _data
    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats['session_id']))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc='sessions (LOC)')
        except ImportError:
            pass
    arrays_by_sid = {}
    for sid, tdf in it:
        try:
            arrays_by_sid[sid] = _session_arrays(tdf)
        except Exception:
            arrays_by_sid[sid] = None
    rows = {r.response_id: lo_localized(arrays_by_sid.get(r.session_id), r.learning_objective, idf, idf_default) for r in feats.itertuples(index=False)}
    out = pd.DataFrame.from_dict(rows, orient='index').reindex(columns=LOC_COLS)
    out.index.name = 'response_id'
    return out.reindex(feats['response_id'])

def lo_alignment(lo_text: str, tok_all: set, tok_stu: set, tok_end: set) -> dict:
    """"""
    lo = _content_tokens(str(lo_text).lower())
    if not lo:
        return {c: 0.0 for c in LO_COLS}
    return {'lo_cov_session': len(lo & tok_all) / len(lo), 'lo_cov_student': len(lo & tok_stu) / len(lo), 'lo_cov_end': len(lo & tok_end) / len(lo)}

def load_lo_idf() -> 'tuple[dict, float] | tuple[None, None]':
    """"""
    import json
    from pathlib import Path
    candidates = [Path(__file__).resolve().parent.parent.parent / 'assets' / 'lo_idf.json', Path(__file__).resolve().parent.parent.parent / 'artifacts' / 'lo_idf.json']
    for p in candidates:
        if p.exists():
            blob = json.loads(p.read_text())
            return (blob['idf'], float(blob['idf_default']))
    return (None, None)
LOC_COLS = ['loc_tutor_confirm_rate', 'loc_tutor_corrective_rate', 'loc_self_expl_rate', 'loc_substantive_rate', 'loc_confusion_rate', 'loc_ends_resolved', 'loc_recency', 'loc_window_share', 'lo_cov_idf', 'loc_elicit_rate', 'loc_gives_answer_rate', 'loc_elicit_vs_tell']
LOC_DELTA_COLS = ['delta_confirm', 'delta_corrective', 'delta_self_expl', 'delta_substantive', 'delta_confusion', 'delta_elicit', 'delta_elicit_vs_tell']
_LOC_DELTA_PAIRS = [('delta_confirm', 'loc_tutor_confirm_rate', 'confirm_rate'), ('delta_corrective', 'loc_tutor_corrective_rate', 'corrective_rate'), ('delta_self_expl', 'loc_self_expl_rate', 'self_explanation_rate'), ('delta_substantive', 'loc_substantive_rate', 'substantive_turn_rate'), ('delta_confusion', 'loc_confusion_rate', 'confusion_rate'), ('delta_elicit', 'loc_elicit_rate', 'tutor_elicit_rate'), ('delta_elicit_vs_tell', 'loc_elicit_vs_tell', 'elicit_to_tell')]

def add_loc_delta_features(X: pd.DataFrame) -> pd.DataFrame:
    """"""
    out = X.copy()
    for name, loc_col, global_col in _LOC_DELTA_PAIRS:
        loc = X[loc_col].fillna(0.0) if loc_col in X.columns else pd.Series(0.0, index=X.index)
        glb = X[global_col].fillna(0.0) if global_col in X.columns else pd.Series(0.0, index=X.index)
        out[name] = loc - glb
    return out

def _loc_empty() -> dict:
    d = {c: 0.0 for c in LOC_COLS}
    d['loc_recency'] = -1.0
    return d

def lo_localized(arrays, lo_text: str, idf: dict, idf_default: float, top_k: int=8, context: int=1) -> dict:
    """"""
    base = _loc_empty()
    if arrays is None:
        return base
    lo = _content_tokens(str(lo_text).lower())
    if not lo:
        return base
    toks, roles, low, wc, n = (arrays['toks'], arrays['roles'], arrays['low'], arrays['wc'], arrays['n'])
    lo_mass = sum((idf.get(t, idf_default) for t in lo))
    if lo_mass <= 0:
        return base
    all_tok = set().union(*toks) if toks else set()
    base['lo_cov_idf'] = sum((idf.get(t, idf_default) for t in lo & all_tok)) / lo_mass
    rel = np.array([sum((idf.get(t, idf_default) for t in toks[i] & lo)) for i in range(n)]) / lo_mass
    relevant = np.where(rel > 0)[0]
    if relevant.size == 0:
        return base
    core = relevant[np.argsort(rel[relevant])[::-1][:top_k]]
    win = sorted({j for i in core for j in range(max(0, i - context), min(n, i + context + 1))})
    base['loc_window_share'] = len(win) / n
    base['loc_recency'] = float(relevant.max()) / n
    win_tut = [i for i in win if roles[i] == 'tutor']
    win_stu = [i for i in win if roles[i] == 'student']
    if win_tut:
        base['loc_tutor_confirm_rate'] = float(np.mean([bool(CONFIRM.search(low[i])) for i in win_tut]))
        base['loc_tutor_corrective_rate'] = float(np.mean([bool(CORRECTIVE.search(low[i])) for i in win_tut]))
        elicit_hits = [bool(ELICIT.search(low[i])) for i in win_tut]
        gives_ans_hits = [bool(GIVES_ANSWER.search(low[i])) for i in win_tut]
        base['loc_elicit_rate'] = float(np.mean(elicit_hits))
        base['loc_gives_answer_rate'] = float(np.mean(gives_ans_hits))
        base['loc_elicit_vs_tell'] = sum(elicit_hits) / (sum(gives_ans_hits) + 1)
    if win_stu:
        base['loc_self_expl_rate'] = float(np.mean([bool(SELF_EXPL.search(low[i])) for i in win_stu]))
        base['loc_substantive_rate'] = float(np.mean([wc[i] > 5 for i in win_stu]))
        base['loc_confusion_rate'] = float(np.mean([bool(CONFUSION.search(low[i])) for i in win_stu]))
        rel_stu = [i for i in relevant if roles[i] == 'student']
        if rel_stu:
            base['loc_ends_resolved'] = float(bool(RESOLUTION.search(low[rel_stu[-1]])))
    return base
INT_COLS = ['repair_complete_rate', 'cfu_affirm_rate', 'answer_attempt_rate', 'answer_confirmed_rate', 'funnel_rate']

def _int_empty() -> dict:
    return {c: 0.0 for c in INT_COLS}

def interaction_stats(arrays) -> dict:
    """"""
    if arrays is None:
        return _int_empty()
    roles, low, wc, n = (arrays['roles'], arrays['low'], arrays['wc'], arrays['n'])
    si, ti = (arrays['si'], arrays['ti'])
    d = _int_empty()
    if len(ti):
        corr_idx = [i for i in ti if CORRECTIVE.search(low[i])]
        if corr_idx:
            conf_idx = [i for i in ti if CONFIRM.search(low[i])]
            completed = sum((any((c > ci for c in conf_idx)) for ci in corr_idx))
            d['repair_complete_rate'] = completed / len(corr_idx)
    affirm = nonaffirm = 0
    for i in ti:
        if CFU.search(low[i]) and i + 1 < n and (roles[i + 1] == 'student'):
            s = low[i + 1]
            w = WORD.findall(s)
            is_ack = bool(w) and len(w) <= 3 and all((t in ACK for t in w))
            if RESOLUTION.search(s) or is_ack:
                affirm += 1
            elif HEDGE.search(s) or CONFUSION.search(s):
                nonaffirm += 1
    if affirm + nonaffirm:
        d['cfu_affirm_rate'] = affirm / (affirm + nonaffirm)
    if len(si):
        attempts = [i for i in si if wc[i] > 5 and (MATH.search(low[i]) or not low[i].rstrip().endswith('?'))]
        if attempts:
            d['answer_attempt_rate'] = len(attempts) / len(si)
            confirmed = 0
            for i in attempts:
                j = i + 1
                while j < n and roles[j] != 'tutor':
                    j += 1
                if j < n and CONFIRM.search(low[j]):
                    confirmed += 1
            d['answer_confirmed_rate'] = confirmed / len(attempts)
    if len(ti):
        tq = [i for i in ti if low[i].rstrip().endswith('?') or ELICIT.search(low[i])]
        if tq:
            answered = 0
            for i in tq:
                j = i + 1
                while j < n:
                    if roles[j] == 'tutor':
                        break
                    if roles[j] == 'student' and wc[j] > 5:
                        answered += 1
                        break
                    j += 1
            d['funnel_rate'] = answered / len(tq)
    return d