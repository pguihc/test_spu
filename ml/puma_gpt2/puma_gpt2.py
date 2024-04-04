import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from typing import Any, Callable, Dict, Optional, Tuple, Union
import jax.nn as jnn
from flax.linen.linear import Array
import jax
import argparse
import json
import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2
from contextlib import contextmanager

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True
Array = Any

PRNGKey = Any

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/puma_gpt2/2pc.json")
args = parser.parse_args()
with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

tokenizer = AutoTokenizer.from_pretrained("gpt2")
pretrained_model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")

# greedy search
# ref: https://huggingface.co/blog/how-to-generate
def text_generation(input_ids, params, token_num=1):
    config = GPT2Config()
    model = FlaxGPT2LMHeadModel(config=config)

    for _ in range(token_num):
        outputs = model(input_ids=input_ids, params=params)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids


def hack_softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = jnp.exp(x) * b

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return nexp / divisor

@contextmanager
def hack_softmax_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    # recover back
    jnn.softmax = raw_softmax

def hack_gelu(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    b0 = x < -4.0
    b1 = x < -1.95
    b2 = x > 3.0
    b3 = b1 ^ b2 ^ True # x in [-1.95, 3.0]
    b4 = b0 ^ b1 # x in [-4, -1.95] 

    # seg1 = a[3] * x^3 + a[2] * x^2 + a[1] * x + a[0]
    # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
    a_coeffs = jnp.array([-0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728])
    b_coeffs = jnp.array([0.008526321541038084,  0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187])
    x2 = jnp.square(x)
    x3 = jnp.multiply(x, x2)
    x4 = jnp.square(x2)
    x6 = jnp.square(x3)

    seg1 = a_coeffs[3] * x3 + a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = b_coeffs[6] * x6 + b_coeffs[4] * x4 + b_coeffs[2] * x2 + b_coeffs[1] * x + b_coeffs[0]

    ret = b2 * x + b4 * seg1 + b3 * seg2

    return ret

@contextmanager
def hack_gelu_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_gelu = jnn.gelu
    jnn.gelu = hack_gelu
    yield
    # recover back
    jnn.gelu = raw_gelu



def run_on_puma():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        'I enjoy walking with my cute dog', return_tensors='jax'
    )
    with hack_softmax_context("hijack jax softmax", enabled = True), hack_gelu_context("hijack jax gelu", enabled=True):
        input_ids = ppd.device("P1")(lambda x: x)(inputs_ids)
        params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
        outputs_ids = ppd.device("SPU")(text_generation,)(input_ids, params)
        outputs_ids = ppd.get(outputs_ids)
    return outputs_ids

print('\n------\nRun on PUMA')
outputs_ids = run_on_puma()
print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
