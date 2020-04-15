def read_lines(file_path):
    files = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            files.append(line.strip())
    return files

def get_probability(target, tokens, token_probs):
    """Get probability of the given target.

    Args:
        target: Justin Martin
        tokens: ['The', ' Archbishop', ' of', ...]
        token_probs: [0.50, 0.49, 0.88, ...] 
    """
    assert len(tokens) == len(token_probs)
    for i, t in enumerate(tokens):
        if len(t) == 0: continue
        prob = 1.0
        t = t.strip()
        if t in target:
            prob = token_probs[i]
            if t == target: return prob
            for ni, (rt, rp) in enumerate(zip(tokens[i+1:], token_probs[i+1:])):
                if t == target: return prob
                elif len(t) < len(target):
                    t += rt
                    prob *= rp
                else:
                    continue
    print('Target ({}) not found!!!'.format(target))
    return -1.0