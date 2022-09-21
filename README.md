# Byte-Offset Tokenizer

Tokenizer for our byte based transformer model. See: [https://huggingface.co/SzegedAI/charmen-electra](https://huggingface.co/SzegedAI/charmen-electra)

# Installation

```
pip install git+https://github.com/szegedai/byte-offset-tokenizer.git
```

# Usage

```python
from byte_offset_tokenizer import ByteOffsetTokenizer

tokenizer = ByteOffsetTokenizer()
tokenizer('Példa mondat!')
```
Output:
```python
{'input_ids': [array([3, 3, 3, ..., 0, 0, 0])], 'attention_mask': [array([ True,  True,  True, ..., False, False, False])], 'token_type_ids': [array([0, 0, 0, ..., 0, 0, 0])]}
```