"""Tokenization utilties for exrepssions."""
import re
from typing import Dict, List, Tuple

from transformers import BertTokenizer

TOKENIZER_PATTERN = "SLG|FTO|TiO2-c|Spiro-MeOTAD|Au|ITO|Ag|TiO2-mp|PCBM-60|PEDOT:PSS|BCP|Al|C60|Carbon|SnO2-c|PTAA|NiO-c|SnO2-np|MoO3|ZnO-c|P3HT|ZnO-np|ZrO2-mp|Cu|LiF|PET|Bphen|Al2O3-mp|NiO-np|TiO2-nw|PEI|TiO2-np|Ca|MoOx|PEN|ZnO-nw|bis-C60|PCBM-70|CuSCN|PFN|PolyTPD|AZO|NiO|PEIE|Rhodamine 101|NiO-mp|NiMgLiO|CuPc|P3CT-Na|Graphene|C60-SAM|PMMA|Ag-nw|Zr(acac)4|AZO-np|Graphene oxide|Nb2O5|ZnO|CuI|IZO|CdS|Pt|SnO2|Ti|Carbon-nt|WO3|Al2O3-c|AgAl|PDMS|TaTm|SnO2-mp|PCBA|SWCNTs|PEAI|P3CT-N|Cu2O|TAPC|C60; PCBM-60|rGO|MgF2|PEDOT|ZnO-mp|F6-TCNNQ; TaTm|ICBA|MoS2|TiO2-nt|Ni|MgO|Carbon-mp|Graphite|TiO2-nanofibers|TiO2|bis‐C60|WOx|VOx|Ag-grid|PFN-Br|CPTA|Ba|[0-9]+|Pb|Br|Cs|Sn|Bi|Te|Sb|Ag|Zn|Sr|Mn|Rb|Cl|Mg|Ba|Cu|Na|Nb|Ni|Eu|Tb|Fe|Li|Pt|Hg|In|Ge|Co|La|Ca|Au|Ti|Al|Sm||A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|t|u|v|w|x|y|z"


class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str) -> None:
        """Constructs a RegexTokenizer.

        Args:
            regex_pattern: regex pattern used for tokenization
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        """Regex tokenization.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens


class PropertyTokenizer:
    """Run a property tokenization."""

    def __init__(self) -> None:
        """Constructs a PropertyTokenizer."""
        self.regex = re.compile(r"\s*(<\w+>)\s*?(\+|-)?(\d+)(\.)?(\d+)?\s*")

    def tokenize(self, text: str) -> List[str]:
        """Tokenization of a property.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = []
        matched = self.regex.match(text)
        if matched:
            property_name, sign, units, dot, decimals = matched.groups()
            tokens = [property_name]
            if sign:
                tokens += [f"_{sign}_"]
            tokens += [
                f"_{number}_{position}_" for position, number in enumerate(units[::-1])
            ][::-1]
            if dot:
                tokens += [f"_{dot}_"]
            if decimals:
                tokens += [
                    f"_{number}_-{position}_"
                    for position, number in enumerate(decimals, 1)
                ]
        return tokens


class PropertyTokenizerSquare(PropertyTokenizer):
    """Run a property tokenization."""

    def __init__(self) -> None:
        """Constructs a PropertyTokenizer."""
        self.regex = re.compile(r"\s*(\[\w+\])\s*?(\+|-)?(\d+)(\.)?(\d+)?\s*")


class CharacterTokenizer:
    def __init__(self) -> None:
        """Constructs a tokenizer that simply splits each character"""
        self.tokenizer = lambda x: list(x)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        return self.tokenizer(text)


class ExpressionTokenizer:
    def __init__(self, expression_tokenizer: str = "|", language: str = "perv") -> None:
        """Constructs an expression tokenizer.

        Args:
            expression_tokenizer (str): Token separating the property. Defaults to '|'.
                Must not occur in the language itself.
            language (str): Identifier for the (chemical) language. Should be either
                'PERV' or 'AAS'.
        """
        self.language = language
        if language == "PERV":
            self.text_tokenizer = RegexTokenizer(regex_pattern=TOKENIZER_PATTERN)
        elif language == "AAS":
            self.text_tokenizer = CharacterTokenizer()
        else:
            raise ValueError(
                f"Unsupported language {language}, choose 'PERV' or 'AAS'."
            )
        self.property_tokenizer = PropertyTokenizer()
        self.expression_separator = expression_tokenizer

    def tokenize(self, text: str) -> List[str]:
        """Tokenize an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        splitted_expression = text.split(self.expression_separator)
        tokens = []
        for property_expression in splitted_expression[:-1]:
            tokens.extend(self.property_tokenizer.tokenize(property_expression))
            tokens.append(self.expression_separator)
        tokens.extend(self.text_tokenizer.tokenize(splitted_expression[-1]))
        return tokens


class ExpressionBertTokenizer(BertTokenizer):
    """
    Constructs a bert-based tokenizer used for the Regression Transformer.

    Args:
        vocab_file: path to a token per line vocabulary file.
    """

    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        pad_even: bool = True,
        language: str = "SMILES",
        **kwargs,
    ) -> None:
        """Constructs an ExpressionTokenizer.

        Args:
            vocab_file: vocabulary file containing tokens.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
            pad_even (bool): Boolean indicating whether sequences of odd length should
                be padded to have an even length. Neede for PLM in XLNet. Defaults to
                True.
            language (str): Identifier for the (chemical) language. Should be either
                'PERV' or 'AAS'.
        """
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=False,
            do_basic_tokenize=True,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        # define tokenization utilities
        self.language = language
        if language == "PERV":
            self.text_tokenizer = RegexTokenizer(regex_pattern=TOKENIZER_PATTERN)
        elif language == "AAS":
            self.text_tokenizer = CharacterTokenizer()
        else:
            raise ValueError(
                f"Unsupported language {language}, choose 'PERV' or 'AAS'."
            )

        self.property_tokenizer = PropertyTokenizer()
        self.expression_separator = "|"
        self.separator_idx = self.vocab[self.expression_separator]
        self.pad_even = pad_even

        # DEPRECATED
        if pad_even:
            self.pad_even_fn = lambda x: x if len(x) % 2 == 0 else x + [self.pad_token]
        else:
            self.pad_even_fn = lambda x: x

    @property
    def vocab_list(self) -> List[str]:
        """List vocabulary tokens.

        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a text representing an expression.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        splitted_expression = text.split(self.expression_separator)
        tokens = []
        for property_expression in splitted_expression[:-1]:
            tokens.extend(self.property_tokenizer.tokenize(property_expression))
            tokens.append(self.expression_separator)
        tokens.extend(self.text_tokenizer.tokenize(splitted_expression[-1]))
        # TODO: remove this hack
        # This is a hack to get around DataCollatorForLanguageModeling requiring even
        # length sequences
        return self.pad_even_fn(tokens)

    def add_padding_tokens(
        self, token_ids: List[int], max_length: int, padding_right: bool = True
    ) -> List[int]:
        """Adds padding tokens to return a sequence of length max_length.

        By default padding tokens are added to the right of the sequence.

        Args:
            token_ids: token indexes.
            max_length: maximum length of the sequence.
            padding_right: whether the sequence is padded on the right. Defaults to True.

        Returns:
            padded sequence of token indexes.
        """
        padding_ids = [self.pad_token_id] * (max_length - len(token_ids))
        if padding_right:
            return token_ids + padding_ids
        else:
            return padding_ids + token_ids

    @staticmethod
    def get_sample_label(mlm_label: List[str], mlm_input: List[str]) -> List[str]:
        """MLM case: Retrieve true sample sequence from mlm label and mlm input.
        NOTE: Also works for PLM.

        Args:
            mlm_label (List[str]): Target sample used in MLM.
            mlm_input (List[str]): MLM input sample.

        Returns:
            List[str]: Sample sequence as part of the dataset
        """

        return [i if el == "[UNK]" else el for el, i in zip(mlm_label, mlm_input)]

    @staticmethod
    def get_sample_prediction(
        mlm_prediction: List[str], mlm_input: List[str]
    ) -> List[str]:
        """MLM case: Retrieve predicted sequence from mlm prediction and mlm input
        NOTE: Also works for PLM.

        Args:
            mlm_label (List[str]): Target sample used in MLM.
            mlm_input (List[str]): MLM input sample.

        Returns:
            List[str]: Sample sequence as part of the dataset
        """
        return [
            i if i not in ["[MASK]"] else o for o, i in zip(mlm_prediction, mlm_input)
        ]

    @staticmethod
    def floating_tokens_to_float(token_ids: List[str]) -> float:
        """Converts tokens representing a float value into a float.
        NOTE: Expects that non-floating tokens are strippped off

        Args:
            token_ids: List of tokens, each representing a float.
                E.g.: ['_0_0_', '_._', '_9_-1_', '_3_-2_', '_1_-3_']

        Returns:
            float: Float representation for the list of tokens.
        """
        try:
            float_string = "".join([token.split("_")[1] for token in token_ids])
            float_value = float(float_string)
        except ValueError:
            float_value = -1
        return float_value

    def aggregate_tokens(
        self, token_ids: List[str], label_mode: bool, cls_first: bool = True
    ) -> Tuple[str, Dict]:
        """Receives tokens of one sample and returns sequence (e.g. SMILES) and
        a dict of properties.

        Args:
            token_ids (List[str]): List of tokens.
            label_mode (bool): Whether the token_ids are labels or predictions.
            cls_first (bool, optional): Whether CLS  token occurres first, default: True

        Returns:
            Tuple[str, Dict]:
                str: SMILES/SELFIES sequence of sample.
                Dict: A dictionary with property names (e.g. 'qed') as key and
                    properties as values.
        """
        edx = min(
            token_ids.index("[SEP]") if "[SEP]" in token_ids else 1000,
            token_ids.index("[PAD]") if "[PAD]" in token_ids else 1000,
        )

        edx = -1 if edx == 1000 else edx

        seq = (
            "".join(token_ids[token_ids.index("|") + 1 : edx])
            if "|" in token_ids
            else "".join(token_ids)
        )
        property_dict = {}
        for idx, t in enumerate(token_ids):
            if t.startswith("<") and t.endswith(">"):
                key = t[1:-1]

                # Convert float
                end_floating_idx = idx + 1
                while token_ids[end_floating_idx].startswith("_"):
                    end_floating_idx += 1

                prop = self.floating_tokens_to_float(
                    token_ids[idx + 1 : end_floating_idx]
                )

                property_dict[key] = prop

        return seq, property_dict

    def to_readable(self, sequence: str) -> str:
        """Safely returns a readable string irrespective of whether the language is
        SMILES, SELFIES or AAS.

        Args:
            sequence (str): A string representing a molecule (either SMILES or SELFIES)
                or amino acid sequence.

        Returns:
            str: A SMILES representing the same molecule.
        """
        if self.language == "PERV":
            return sequence
        elif self.language == "AAS":
            return sequence
        else:
            raise AttributeError(f"Unknown language {self.language}")
