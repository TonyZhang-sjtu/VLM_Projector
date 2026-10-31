"""Separate exact sample/token validation from historical floating-point replay."""
import math


def compare_ce_history(records, historical):
    if len(records) != historical['samples']:
        raise ValueError('CE historical sample count mismatch')
    groups = {}
    for row in records:
        if (type(row['answer_tokens']) is not int or row['answer_tokens'] <= 0
                or not math.isfinite(row['loss_sum']) or row['loss_sum'] < 0):
            raise ValueError('Invalid CE numerator/denominator')
        groups.setdefault(row['cohort'], []).append(row)
    if set(groups) != set(historical['cohorts']):
        raise ValueError('CE historical cohort coverage mismatch')
    total = math.fsum(r['loss_sum'] for r in records)
    tokens = sum(r['answer_tokens'] for r in records)
    if tokens != historical['answer_tokens']:
        raise ValueError('CE historical token count mismatch')
    def comparison(actual, expected):
        return dict(recomputed=actual, historical=expected, difference=actual-expected,
                    matches=math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5))
    cohorts = {}
    for name, rows in groups.items():
        expected = historical['cohorts'][name]
        count = sum(r['answer_tokens'] for r in rows)
        if count != expected['answer_tokens'] or len(rows) != expected['samples']:
            raise ValueError(f'CE historical cohort sample/token mismatch: {name}')
        loss = math.fsum(r['loss_sum'] for r in rows)
        cohorts[name] = dict(comparison(loss/count, expected['answer_ce']),
                            samples=len(rows), answer_tokens=count, loss_sum=loss,
                            historical_loss_sum=expected['loss_sum'])
    overall = comparison(total/tokens, historical['answer_ce'])
    return dict(overall=overall, cohorts=cohorts, loss_sum=total, answer_tokens=tokens,
                all_match=overall['matches'] and all(c['matches'] for c in cohorts.values()))


def require_historical_match(comparison, policy):
    if policy not in ('strict', 'recompute'):
        raise ValueError('Unknown CE replay policy')
    if policy == 'strict' and not comparison['all_match']:
        raise ValueError(f'CE replay differs from historical result: {comparison["overall"]}; '
                         'see historical_comparison.json for cohort differences')
