import os
from functools import partial

# Let VLMEvalKit assign GPUs before importing the model.
from VLMEvalKit import run
from main import reproduce
from vlmeval_adapter import IPMoEVLMEval

if __name__ == '__main__':
    seed = int(os.environ['PYTHONHASHSEED'])
    reproduce(seed)
    run.supported_VLM['IPMoE'] = partial(IPMoEVLMEval, checkpoint_path = f'checkpoint/seed_{seed}/checkpoint.pt')

    run.load_env()
    run.main()
