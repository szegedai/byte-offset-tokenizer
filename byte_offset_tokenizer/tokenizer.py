import numpy as np
import torch


class ByteOffsetTokenizer:
    def __init__(self):
        self.mappings = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[SEP]": 2,
            "[CLS]": 3,
            "[UNK]": 4,
        }
        self.mappings_revers = {self.mappings[k]: k for k in self.mappings}
        self.spacial_ids = [self.mappings[k] for k in self.mappings]
        self.spacial_tokens = [k for k in self.mappings]
        self.vocab_size = 256 + len(self.spacial_ids)
        self.mask_token_id = self.mappings["[MASK]"]
        self.pad_token_id = self.mappings["[PAD]"]
        self.is_fast = True

    def __len__(self):
        return self.vocab_size

    def __call__(self, text1, text2=None, truncation=True, max_length=1024, ds_factor=4,
                 return_special_tokens_mask=False, return_offsets_mapping=False, return_tensors=None, **kwargs):
        return self.tokenize(text1, text2, truncation, max_length, ds_factor, return_special_tokens_mask,
                             return_offsets_mapping, return_tensors)

    def encode(self, text1, text2=None, truncation=True, max_length=1024, ds_factor=4,
               return_special_tokens_mask=False, return_offsets_mapping=False, return_tensors=None):
        return self(text1, text2, truncation, max_length, ds_factor, return_special_tokens_mask,
                    return_offsets_mapping, return_tensors)['input_ids']

    def tokenize(self, text1, text2=None, truncation=True, max_length=1024, ds_factor=4,
                 return_special_tokens_mask=False, return_offsets_mapping=False, return_tensors=None):
        was_str = False
        if type(text1) is str:
            text1 = [text1]
            was_str = True
        if type(text2) is str:
            text2 = [text2]

        input_ids = []
        attention_mask = []
        type_embeddings = []
        spm = []
        offset_mapping = []
        if text2 is None:
            text2 = [''] * len(text1)
        for t1, t2 in zip(text1, text2):
            t1 = bytearray(t1, encoding="utf8")
            t2 = bytearray(t2, encoding="utf8")

            t1_length = len(t1)
            t2_length = len(t2)

            out = np.zeros(max_length, dtype=np.int)
            attention = np.zeros(max_length, dtype=np.bool)
            type_embedding = np.zeros(max_length, dtype=np.int)
            special_token_mapping = np.zeros(max_length, dtype=np.bool)
            offset = [(0, 0) for _ in range(ds_factor)]
            offset_start = 0
            offset_end = 0

            type_embedding[t1_length + ds_factor + 1:t1_length + t2_length + ds_factor + 1] = 1

            out[:ds_factor] = [self.mappings['[CLS]'] for _ in range(ds_factor)]
            special_token_mapping[:ds_factor] = True

            if t1_length + ds_factor < max_length:
                out[t1_length + ds_factor] = self.mappings['[SEP]']
                special_token_mapping[t1_length + ds_factor] = True
            else:
                out[-1] = self.mappings['[SEP]']
                special_token_mapping[-1] = True

            attention[:ds_factor] = True
            if t1_length + ds_factor < max_length:
                attention[t1_length + 1] = True
            else:
                attention[-1] = True
            for i, b in enumerate(t1):
                k = i + ds_factor
                if truncation and k >= max_length:
                    break
                out[k] = b + len(self.spacial_ids)
                if b != 0x20:
                    offset_end += 1
                else:
                    offset.append((offset_start, offset_end))
                    offset_end = offset_end + 1
                    offset_start = offset_end
                attention[k] = True

            for i, b in enumerate(t2):
                k = t1_length + i + ds_factor + 1
                if truncation and k >= max_length:
                    break
                out[k] = b + len(self.spacial_ids)
                attention[k] = True

            if t2_length > 0:
                if t1_length + t2_length + ds_factor + 1 < max_length:
                    out[t1_length + t2_length + ds_factor + 1] = self.mappings['[SEP]']
                    special_token_mapping[t1_length + t2_length + ds_factor + 1] = True
                    attention[t1_length + t2_length + ds_factor + 1] = True
                else:
                    out[-1] = self.mappings['[SEP]']
                    special_token_mapping[-1] = True

            input_ids.append(out)
            attention_mask.append(attention)
            type_embeddings.append(type_embedding)
            spm.append(special_token_mapping)
            offset_mapping.append(offset)
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_embeddings
        }
        if return_special_tokens_mask:
            output['special_tokens_mask'] = spm
        if return_offsets_mapping:
            output['offset_mapping'] = offset_mapping

        if return_tensors is not None:
            for key in output:
                item = torch.tensor(output[key])
                output[key] = item
        return output

    def convert_ids_to_tokens(self, x):
        return x - len(self.spacial_ids)
