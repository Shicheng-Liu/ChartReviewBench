"""Exercise OpenRouter serialization/accounting against a canned SDK client."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from openai.types.chat import ChatCompletion
from chartsandbox.providers.openrouter_provider import OpenRouterProvider
from chartsandbox.providers import resolve, image_block, user, assistant, tool_result_block, text_block

class Client:
    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kw):
        self.requests.append(kw)
        judge = 'response_format' in kw
        msg = {'role': 'assistant', 'content': '{"passed":true,"score":1.0,"detail":"readable"}' if judge else '<decision>stop</decision>',
               'reasoning': 'checked the plot',
               'reasoning_details': [{'type':'reasoning.encrypted','data':'signed-test-value','format':'test'}]}
        if kw.get('tools'):
            msg['tool_calls'] = [{'id':'call1','type':'function','function':{'name':'finish','arguments':'{}'}}]
        return ChatCompletion.model_validate({'id':'gen-test','object':'chat.completion','created':0,'model':'vendor/model',
            'provider':'test-provider','choices':[{'index':0,'finish_reason':'stop','message':msg}],
            'usage':{'prompt_tokens':100,'completion_tokens':40,'total_tokens':140,
                     'completion_tokens_details':{'reasoning_tokens':30},
                     'prompt_tokens_details':{'cached_tokens':60,'cache_write_tokens':10},'cost':0.001}})

assert resolve('openrouter:vendor/model:variant') == ('openrouter','vendor/model:variant')
client = Client()
p = OpenRouterProvider('vendor/model',client=client,effort='high')
messages = [user([text_block('inspect'),image_block('image/png','dGVzdA==')])]
turn = p.complete('system',messages,[])
assert 'tools' not in client.requests[0]
assert client.requests[0]['extra_body']['reasoning'] == {'effort':'high'}
assert client.requests[0]['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
assert turn.thinking == 'checked the plot'
messages.append(assistant(turn.blocks,turn.raw,p.name))
tools = [{'name':'finish','description':'done','input_schema':{'type':'object','properties':{}}}]
turn2 = p.complete('system',messages,tools)
assert client.requests[-1]['messages'][-1]['reasoning_details'] == turn.raw['reasoning_details']
assert client.requests[-1]['parallel_tool_calls'] is False and turn2.tool_calls[0]['id']=='call1'
messages.extend([assistant(turn2.blocks,turn2.raw,p.name), user([tool_result_block('call1',[text_block('ok'),image_block('image/png','dGVzdA==')])])])
p.complete('system',messages,tools)
assert any(m['role']=='tool' and m['tool_call_id']=='call1' for m in client.requests[-1]['messages'])
mark = p.mark()
assert p.judge('legible',[{'mime':'image/png','b64':'dGVzdA=='}]).passed
assert client.requests[-1]['response_format']['type']=='json_schema'
assert 'guided_json' not in str(client.requests[-1])
assert p.episode_stats(mark)['calls']==1
assert p.stats()['usage']['input_tokens']==400
assert p.stats()['usage']['output_tokens']==160
assert p.stats()['usage']['reasoning_tokens']==120
assert p.stats()['usage']['cache_read_input_tokens']==240
assert p.stats()['usage']['cache_write_input_tokens']==40
assert len(p.stats()['responses'])==4
print('OpenRouter: XML, tools, image bytes, reasoning replay, judge schema and token accounting passed (no network)')
