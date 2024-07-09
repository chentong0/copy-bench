import pickle
import os
import time
import json

class LM(object):

    def __init__(self, cache_file, save_interval=100):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.save_interval = save_interval
        self.model = None
        self.add_n = 0

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    @staticmethod
    def get_cache_key(prompt, sample_idx):
        prompt_serialized = prompt if isinstance(prompt, str) else json.dumps(prompt)
        return f"{prompt_serialized}_{sample_idx}"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, **kwargs):
        if self.model is None:
            self.load_model()
        
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        
        # prompt = prompt.strip() # it's important not to end with a whitespace
        if isinstance(prompt, str):
            assert not prompt.endswith(" ")
            # assert not prompt.endswith("\n")
        
        cache_key = self.get_cache_key(prompt, sample_idx)
        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        if isinstance(prompt, str) and prompt.endswith(" True or False?\nAnswer:"):
            raise NotImplementedError("This is a special case that is not implemented yet.")
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        else:
            generated = self._generate(prompt, **kwargs)

        self.cache_dict[cache_key] = generated
        self.add_n += 1
        return generated

    def save_cache(self):
        if self.cache_file is None:
            return

        if self.add_n == 0:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if self.cache_file is None:
            return {}
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache