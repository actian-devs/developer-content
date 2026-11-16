"""runtime-validator-v2.0.0: exact, case-sensitive order-identifier membership."""
import re

VERSION='runtime-validator-v2.0.0'
ORDER=re.compile(r'(?<![\w-])ORD-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?![\w-])')

def validate_order_membership(result,message):
    result=dict(result);result['runtime_errors']=list(result.get('runtime_errors',[]))
    parsed=result.get('parsed')
    if parsed and parsed.get('order_id') is not None:
        if parsed['order_id'] not in {m.group() for m in ORDER.finditer(message)}:
            result['runtime_errors'].append('Returned order_id does not exactly occur as a complete order identifier in the input')
            result['runtime_check_pass']=False
    result['validator_version']=VERSION
    return result
