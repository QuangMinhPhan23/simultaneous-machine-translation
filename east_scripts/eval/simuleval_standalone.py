"""Adaptive read/write SiMT inference and scoring (BLEU, spBLEU, chrF++, COMET, optional BERTScore).

Input: a model path and a test JSON file. Output: prediction.json and results.json.
Same evaluation as simuleval.py but with no llamafactory dependency, so it only needs
torch, transformers, sacrebleu and comet.
Set --eot_token to match the model family, e.g. <end_of_turn> for Gemma-3 / Nile-Chat.
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu
from comet import load_from_checkpoint
from statistics import mean

from latency import compute_delays, AverageLagging, LengthAdaptiveAverageLagging

EOT_TOKEN_DEFAULT = "<|eot_id|>"  # Llama-3's turn-end token


def tokenize_chinese(text, tokenizer):
    """Split Chinese text into readable units, since it has no spaces to split on.

    Token ids are collected one at a time and only flushed once they decode without the
    replacement character, so a unit is never cut in the middle of a character."""
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    idx = 0
    tok_ids = []
    tokens = []
    while idx < len(input_ids):
        tok_ids.append(input_ids[idx])
        token = tokenizer.decode(tok_ids)
        if "�" not in token:
            tokens.append(token)
            tok_ids = []
        idx += 1
    return tokens


class SimulInference:
    """Runs simultaneous translation over a test set and scores the output.

    The model alternates between two phases: READ, where it takes one more source word, and
    WRITE, where it emits a piece of translation. It signals the switch itself with the
    <|end-of-read|> and <|end-of-write|> tokens."""

    def __init__(self, args):
        self.args = args

        self.load_tokenizer_and_model(self.args.model_path)
        self.gen_kwargs = self.prepare_gen_kwargs(args)
        self.set_special_tokens()

        self.test_data = self.load_eval_datasets(self.args.data_path)
        self.instruction = "You are a professional simultaneous interpreter, your task is to translate the following {src_lang} text into {tgt_lang} with {latency} latency."
        self.latency = self.args.latency

        self.predictions = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_tokenizer_and_model(self, model_path):
        """Load the checkpoint in bfloat16, move it to the GPU if there is one, and set eval mode."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        # Use the turn-end token as EOS, matching what llamafactory does.
        self.tokenizer.add_special_tokens({"eos_token": self.args.eot_token})

    def prepare_gen_kwargs(self, args):
        """Base generation settings shared by both phases: no sampling, so runs are reproducible.

        Both the tokenizer's EOS and the turn-end token are treated as stop tokens."""
        gen_kwargs = {}
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
        gen_kwargs["num_beams"] = args.num_beams
        gen_kwargs["eos_token_id"] = list(set([self.tokenizer.eos_token_id]))
        gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        gen_kwargs["eos_token_id"].append(self.tokenizer.convert_tokens_to_ids(self.args.eot_token))
        gen_kwargs["eos_token_id"] = list(set(gen_kwargs["eos_token_id"]))
        return gen_kwargs

    def prepare_read_kwargs(self):
        """Settings for a READ step: give the model exactly one token to answer one question,
        keep reading or start writing? Write markers are blocked, so the only thing it can emit
        to switch phase is <|end-of-read|>."""
        self.gen_kwargs["suppress_tokens"] = self.read_suppress_tok_ids
        self.gen_kwargs["eos_token_id"] = self.read_eos_tok_ids
        self.gen_kwargs["num_beams"] = 1
        self.gen_kwargs["max_new_tokens"] = 1

    def prepare_write_kwargs(self, read_tok_num):
        """Settings for a WRITE step: generate a translation chunk until <|end-of-write|>.

        <|end-of-read|> is blocked here, and the token budget grows with how much source was
        just read, capped by --max_new_tokens."""
        self.gen_kwargs["num_beams"] = self.args.num_beams
        self.gen_kwargs["suppress_tokens"] = self.write_suppress_tok_ids
        self.gen_kwargs["eos_token_id"] = self.write_eos_tok_ids
        max_new_tokens = (read_tok_num + 25) * 2
        self.gen_kwargs["max_new_tokens"] = min(self.args.max_new_tokens, max_new_tokens)

    def set_special_tokens(self):
        """Look up the ids of the read/write markers and work out which ids each phase allows.

        Each phase blocks the other phase's marker, so the model cannot skip a step."""
        self.eor_token = "<|end-of-read|>"
        self.eow_token = "<|end-of-write|>"

        eor_tok_id = self.tokenizer(self.eor_token, add_special_tokens=False).input_ids
        eow_tok_id = self.tokenizer(self.eow_token, add_special_tokens=False).input_ids

        self.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id == list else [self.tokenizer.bos_token_id]

        self.read_eos_tok_ids = eor_tok_id
        self.write_eos_tok_ids = self.gen_kwargs["eos_token_id"] + eow_tok_id

        self.read_suppress_tok_ids = self.gen_kwargs["eos_token_id"] + eow_tok_id + bos_token_id
        self.write_suppress_tok_ids = eor_tok_id + bos_token_id

    def load_eval_datasets(self, data_path):
        """Read the test JSON: a list of {source, reference, src_lang, tgt_lang} rows."""
        with open(data_path, "r") as f:
            data = json.load(f)
        if self.args.max_examples:
            data = data[: self.args.max_examples]
        return data

    def eval_instance_with_beam_search(self, index, sample):
        """Translate one sentence with the adaptive read/write loop, using beam search to write.

        The source is fed in one word at a time. After each word the model gets a single token to
        say whether it wants to keep reading or start writing; when it writes, it produces a chunk
        and closes it with <|end-of-write|>, then goes back to reading. Everything is appended to
        one growing string, and a KV cache avoids recomputing the prefix each step."""
        src_text = sample["source"]
        ref = sample["reference"]
        src_lang = sample["src_lang"]
        tgt_lang = sample["tgt_lang"]

        instruction = self.instruction.format(src_lang=src_lang, tgt_lang=tgt_lang, latency=self.latency)
        messages = [{"role": "user", "content": instruction}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        past_key_values = None
        prev_input_len = 0

        # The source is revealed one unit at a time: characters for Chinese, whitespace words
        # otherwise. This list is what "how much has been read so far" is measured in.
        if src_lang == "Chinese":
            src_tokens = tokenize_chinese(src_text, self.tokenizer)
        else:
            src_tokens = src_text.split()

        input_text = prompt
        is_read = True

        idx = 0
        read_tok_num = 0
        preds = []
        read_chunk = []
        read_contents = []

        # Loop until the source is used up AND no write is still in progress.
        while idx < len(src_tokens) or (not is_read):
            prev_key_values = past_key_values
            if is_read:
                # READ: reveal the next source word and let the model answer with one token.
                input_token = src_tokens[idx]
                if idx == 0 or src_lang == "Chinese":
                    input_text = f"{input_text}{input_token}"
                else:
                    input_text = f"{input_text} {input_token}"

                self.prepare_read_kwargs()
                idx += 1
                read_tok_num += 1
                read_chunk.append(input_token)
            else:
                # WRITE: close the run of source words just read and store it. read_contents is
                # what the latency metrics use to know how much input each output chunk saw.
                if src_lang == "Chinese":
                    read_contents.append("".join(read_chunk))
                else:
                    read_contents.append(" ".join(read_chunk))

                read_chunk = []
                self.prepare_write_kwargs(read_tok_num)
                num_beams = self.gen_kwargs["num_beams"]

                # Beam search runs num_beams copies of the sequence, so the cached keys and
                # values have to be repeated to match.
                if past_key_values is not None:
                    past_key_values = tuple((k.repeat(num_beams, 1, 1, 1), v.repeat(num_beams, 1, 1, 1)) for k, v in past_key_values)

            model_inputs = self.tokenizer([input_text], add_special_tokens=False, return_tensors="pt").to(self.device)

            curr_input_len = model_inputs.input_ids[0].size(0)

            # Adding a word can re-tokenize the tail and leave the text no longer than before.
            # Trim the cache by the difference so cache length and input length stay aligned.
            if curr_input_len - prev_input_len < 1:
                less_token = 1 - (curr_input_len - prev_input_len)
                past_key_values = tuple((k[:, :, :-less_token, :], v[:, :, :-less_token, :]) for k, v in past_key_values)

            prev_input_len = curr_input_len

            model_output = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                past_key_values=past_key_values,
                **self.gen_kwargs,
            )

            generated_ids = model_output.sequences
            past_key_values = model_output.past_key_values

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # A write step's cache belongs to the beams, not to the single sequence we keep, so
            # roll it back to the cache from before the step.
            if not is_read:
                past_key_values = prev_key_values

            # Treat a plain EOS as an end-of-write, so both endings are handled the same way.
            response = response.replace(self.eos_token, self.eow_token)

            # Decide what just happened and what the next step should be.
            if is_read and self.eor_token in response:
                # The model asked to stop reading: switch to writing.
                is_read = False
                input_text = f"{input_text}{response}"
            elif not is_read and (self.eow_token in response or generated_ids[0].size(0) >= self.gen_kwargs["max_new_tokens"]):
                # A translation chunk finished (or hit its token budget): keep it and read again.
                is_read = True
                read_tok_num = 0
                hypo = response.rstrip().replace(self.eow_token, "")
                if self.eow_token not in response:
                    response = f"{response}{self.eow_token}"
                input_text = f"{input_text}{response}"
                preds.append(hypo)
            elif idx >= len(src_tokens):
                # No source left, so force the switch to writing whatever is still owed.
                is_read = False
                input_text = f"{input_text}{self.eor_token}"
                past_key_values = prev_key_values
            elif is_read and len(generated_ids[0]) > 1:
                # The read step emitted more than the one token we allowed, so its cache no
                # longer matches input_text; drop it.
                past_key_values = prev_key_values
            elif is_read and self.args.document_level and input_text[-1] in "。？！.!?" and read_tok_num >= 20:
                # Document mode: force a write at a sentence boundary once enough has been read.
                is_read = False
                input_text = f"{input_text}{self.eor_token}"

        # `output` keeps the raw text with the read/write markers, so the decision pattern can be
        # inspected later; `translation` is just the written chunks joined together.
        output = input_text[len(prompt):].strip()
        translation = "".join(preds)

        self.predictions.append(
            {
                "index": index,
                "source": src_text,
                "reference": ref,
                "prediction": translation,
                "output": output,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "read_contents": read_contents,
                "hypo": preds,
            }
        )

    def eval_instance_with_greedy_search(self, index, sample):
        """Same read/write loop as the beam-search version, but writing picks the single most
        likely next token each time.

        With one beam there is nothing to duplicate or roll back, so the KV cache is simply
        carried forward and the extra cache bookkeeping is not needed."""
        src_text = sample["source"]
        ref = sample["reference"]
        src_lang = sample["src_lang"]
        tgt_lang = sample["tgt_lang"]

        instruction = self.instruction.format(src_lang=src_lang, tgt_lang=tgt_lang, latency=self.latency)
        messages = [{"role": "user", "content": instruction}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        past_key_values = None

        if src_lang == "Chinese":
            src_tokens = tokenize_chinese(src_text, self.tokenizer)
        else:
            src_tokens = src_text.split()

        input_text = prompt
        is_read = True

        idx = 0
        preds = []
        read_tok_num = 0
        read_contents = []
        read_chunk = []

        # Loop until the source is used up AND no write is still in progress.
        while idx < len(src_tokens) or (not is_read):
            if is_read:
                # READ: reveal the next source word and let the model answer with one token.
                input_token = src_tokens[idx]
                if idx == 0 or src_lang == "Chinese":
                    input_text = f"{input_text}{input_token}"
                else:
                    input_text = f"{input_text} {input_token}"

                self.prepare_read_kwargs()
                idx += 1
                read_tok_num += 1
                read_chunk.append(input_token)
            else:
                # WRITE: close the run of source words just read and store it for the latency
                # metrics, then set up a full generation step.
                if src_lang == "Chinese":
                    read_contents.append("".join(read_chunk))
                else:
                    read_contents.append(" ".join(read_chunk))

                read_chunk = []
                self.prepare_write_kwargs(read_tok_num)

            model_inputs = self.tokenizer([input_text], add_special_tokens=False, return_tensors="pt").to(self.device)

            model_output = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                past_key_values=past_key_values,
                **self.gen_kwargs,
            )

            generated_ids = model_output.sequences
            past_key_values = model_output.past_key_values

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Treat a plain EOS as an end-of-write, so both endings are handled the same way.
            response = response.replace(self.eos_token, self.eow_token)

            # Decide what just happened and what the next step should be.
            if is_read and self.eor_token in response:
                # The model asked to stop reading: switch to writing.
                is_read = False
                input_text = f"{input_text}{response}"
            elif not is_read and (self.eow_token in response or generated_ids[0].size(0) >= self.gen_kwargs["max_new_tokens"]):
                # A translation chunk finished (or hit its token budget): keep it and read again.
                hypo = response.rstrip().replace(self.eow_token, "")
                if self.eow_token not in response:
                    response = f"{response}{self.eow_token}"
                input_text = f"{input_text}{response}"
                preds.append(hypo)
                is_read = True
                read_tok_num = 0
            elif idx >= len(src_tokens):
                # No source left, so force the switch to writing whatever is still owed.
                is_read = False
                input_text = f"{input_text}{self.eor_token}"
            elif is_read and self.args.document_level and input_text[-1] in "。？！.!?" and read_tok_num >= 20:
                # Document mode: force a write at a sentence boundary once enough has been read.
                is_read = False
                input_text = f"{input_text}{self.eor_token}"

        # `output` keeps the raw text with the read/write markers, so the decision pattern can be
        # inspected later; `translation` is just the written chunks joined together.
        output = input_text[len(prompt):].strip()
        translation = "".join(preds)

        self.predictions.append(
            {
                "index": index,
                "source": src_text,
                "reference": ref,
                "prediction": translation,
                "output": output,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "read_contents": read_contents,
                "hypo": preds,
            }
        )

    def cal_scores(self):
        """Score quality and latency on the CPU, and write per-sentence numbers back onto
        each prediction.

        Quality: BLEU counts matching word n-grams against the reference; chrF++ does the same
        on character n-grams plus word bigrams, which is fairer to dialectal spelling variation.
        Latency: AL (Average Lagging) is roughly how many source words behind a live speaker the
        system is when it writes, and LAAL is the same idea but not penalised when the output
        length differs from the reference. For both, lower means less delay."""
        hypos = []
        refs = []
        results = {}
        ALs = []
        LAALs = []

        for prediction in self.predictions:
            translation = prediction["prediction"]
            ref = prediction["reference"]
            src_lang = prediction["src_lang"]
            tgt_lang = prediction["tgt_lang"]
            read_contents = prediction["read_contents"]
            hypo = prediction["hypo"]

            hypos.append(translation)
            refs.append(ref)

            # Chinese is counted in characters and needs sacrebleu's "zh" tokenizer; other
            # languages use the default "13a" word tokenizer.
            if tgt_lang == "Chinese":
                tok = "zh"
                ref_len = len(list(ref))
            else:
                tok = "13a"
                ref_len = len(ref.split())

            if src_lang == "Chinese":
                src_len = len(list(prediction["source"]))
            else:
                src_len = len(prediction["source"].split())

            bleu = sacrebleu.sentence_bleu(translation, [ref], tokenize=tok).score

            # delays[i] = how many source words had been read when output word i was produced.
            delays, _ = compute_delays(read_contents, hypo, src_lang, tgt_lang)
            AL = AverageLagging(delays, src_len, ref_len)
            LAAL = LengthAdaptiveAverageLagging(delays, src_len, ref_len)

            ALs.append(AL)
            LAALs.append(LAAL)

            prediction["delays"] = str(delays)
            prediction["BLEU"] = bleu
            prediction["AL"] = AL
            prediction["LAAL"] = LAAL

        # Corpus scores are computed over the whole test set at once, which is the number
        # reported in the results table (not the mean of the per-sentence BLEUs above).
        tok = "zh" if self.predictions[0]["tgt_lang"] == "Chinese" else "13a"

        bleu_score = sacrebleu.corpus_bleu(hypos, [refs], tokenize=tok).score

        # All three surface metrics at once (CPU only), so no separate rescoring
        # pass is needed later. spBLEU uses the flores200 SentencePiece tokenizer.
        results["BLEU"] = bleu_score
        try:
            results["spBLEU"] = sacrebleu.corpus_bleu(
                hypos, [refs], tokenize="flores200"
            ).score
        except Exception as e:  # noqa: BLE001, the flores200 model may fail to download
            print(f"WARNING: spBLEU (flores200) failed ({e}); other metrics unaffected.")
            results["spBLEU"] = None
        results["chrF++"] = sacrebleu.corpus_chrf(hypos, [refs], word_order=2).score

        results["AL"] = mean(ALs)
        results["LAAL"] = mean(LAALs)

        return results

    def compute_neural_metrics(self):
        """Run the neural metrics (COMET, BERTScore, optional BLEURT) and return
        corpus scores, all scaled x100. Per-sentence values are written back onto
        each prediction. Each metric is guarded on its own so one failure does not
        lose the others."""
        out = {}
        data = [
            {"src": item["source"], "mt": item["prediction"], "ref": item["reference"]}
            for item in self.predictions
        ]
        gpus = 1 if torch.cuda.is_available() else 0

        # COMET is a trained neural scorer: it reads source, output and reference together and
        # predicts a human-like quality score, so it can reward a correct paraphrase that BLEU
        # would miss. Each checkpoint is freed before the next one is loaded.
        comet_ckpts = [self.args.comet_ckpt_path] + list(self.args.extra_comet_ckpts or [])
        for i, ckpt in enumerate(comet_ckpts):
            # Extras get their folder name appended so they do not overwrite the main COMET.
            name = "COMET" if i == 0 else f"COMET::{os.path.basename(os.path.dirname(ckpt)) or ckpt}"
            try:
                model = load_from_checkpoint(ckpt, reload_hparams=True)
                pred = model.predict(data, batch_size=256, gpus=gpus)
                for idx, s in enumerate(pred.scores):
                    self.predictions[idx][name] = s * 100
                out[name] = pred.system_score * 100
                del model
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: {name} scoring failed ({e}); other metrics unaffected.")

        # BERTScore compares the contextual embeddings of output and reference word by word and
        # reports the F1 of the best matches, so wording differences hurt it less than BLEU.
        if self.args.bertscore_model:
            try:
                from bert_score import score as bertscore_fn  # imported here so it stays optional
                hyps = [p["prediction"] for p in self.predictions]
                refs_bs = [p["reference"] for p in self.predictions]
                _, _, f1 = bertscore_fn(
                    hyps, refs_bs,
                    model_type=self.args.bertscore_model,
                    verbose=False,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                f1 = (f1 * 100).tolist()
                for idx, v in enumerate(f1):
                    self.predictions[idx]["BERTScore"] = v
                out["BERTScore"] = mean(f1)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: BERTScore failed ({e}); other metrics unaffected.")

        # BLEURT is off by default because it pulls in TensorFlow.
        if self.args.bleurt_ckpt_path:
            try:
                from bleurt import score
                scorer = score.BleurtScorer(self.args.bleurt_ckpt_path)
                vals = []
                for p in self.predictions:
                    b = scorer.score(references=[p["reference"]],
                                     candidates=[p["prediction"]])[0] * 100
                    p["BLEURT"] = b
                    vals.append(b)
                out["BLEURT"] = mean(vals)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: BLEURT failed ({e}); other metrics unaffected.")

        return out

    def save_results(self, results, output_dir):
        """Write prediction.json (every sentence) and results.json (the corpus scores)."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_file = os.path.join(output_dir, "prediction.json")
        with open(prediction_file, "w", encoding="utf8") as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=4)

        result_file = os.path.join(output_dir, "results.json")
        with open(result_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def simul_eval(self):
        """Translate every test sentence, then score the run and save it.

        The translation model is moved off the GPU first, so the metric models have room."""
        if self.args.num_beams == 1:
            eval_instance_func = self.eval_instance_with_greedy_search
        elif self.args.num_beams > 1:
            eval_instance_func = self.eval_instance_with_beam_search
        else:
            raise ValueError("num_beams must be greater then or equal to 1.")

        with torch.no_grad():
            for index, sample in tqdm(enumerate(self.test_data), total=len(self.test_data)):
                eval_instance_func(index, sample)

        self.model.cpu()
        torch.cuda.empty_cache()

        results = self.cal_scores()

        # Save before the neural metrics so a scoring crash never throws away
        # the generation, which is the expensive part.
        self.save_results(results, self.args.output_dir)

        try:
            results.update(self.compute_neural_metrics())
            self.save_results(results, self.args.output_dir)
        except Exception as e:
            print(f"WARNING: neural metric scoring failed ({e}); BLEU/spBLEU/chrF++/AL/LAAL + predictions were already saved.")


def load_infer_args():
    """Command-line options for one eval run: which model, which test file, which latency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--latency", type=str, required=True)
    parser.add_argument("--eot_token", type=str, default=EOT_TOKEN_DEFAULT,
                         help="Turn-end token, e.g. <|eot_id|> for Llama-3 or <end_of_turn> for Gemma-3")
    parser.add_argument("--bleurt_ckpt_path", type=str, default=None,
                         help="Optional; omit to skip BLEURT and its TensorFlow dependency")
    parser.add_argument("--comet_ckpt_path", type=str, required=True)
    parser.add_argument("--extra_comet_ckpts", type=str, nargs="*", default=None,
                         help="Extra COMET checkpoints to score, each reported as COMET::<folder>")
    parser.add_argument("--bertscore_model", type=str, default="bert-base-multilingual-cased",
                         help="HF model for BERTScore F1; pass an empty string to skip BERTScore")
    parser.add_argument("--document_level", type=bool, default=False)
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Cap number of test examples, useful for a quick sanity check")
    return parser.parse_args()


def run_simuleval():
    """Entry point: parse the arguments, then run inference and scoring."""
    args = load_infer_args()
    infer = SimulInference(args)
    infer.simul_eval()


if __name__ == "__main__":
    run_simuleval()
