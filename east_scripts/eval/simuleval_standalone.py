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
        self.gen_kwargs["suppress_tokens"] = self.read_suppress_tok_ids
        self.gen_kwargs["eos_token_id"] = self.read_eos_tok_ids
        self.gen_kwargs["num_beams"] = 1
        self.gen_kwargs["max_new_tokens"] = 1

    def prepare_write_kwargs(self, read_tok_num):
        self.gen_kwargs["num_beams"] = self.args.num_beams
        self.gen_kwargs["suppress_tokens"] = self.write_suppress_tok_ids
        self.gen_kwargs["eos_token_id"] = self.write_eos_tok_ids
        max_new_tokens = (read_tok_num + 25) * 2
        self.gen_kwargs["max_new_tokens"] = min(self.args.max_new_tokens, max_new_tokens)

    def set_special_tokens(self):
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
        with open(data_path, "r") as f:
            data = json.load(f)
        if self.args.max_examples:
            data = data[: self.args.max_examples]
        return data

    def eval_instance_with_beam_search(self, index, sample):
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

        while idx < len(src_tokens) or (not is_read):
            prev_key_values = past_key_values
            if is_read:
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
                if src_lang == "Chinese":
                    read_contents.append("".join(read_chunk))
                else:
                    read_contents.append(" ".join(read_chunk))

                read_chunk = []
                self.prepare_write_kwargs(read_tok_num)
                num_beams = self.gen_kwargs["num_beams"]

                if past_key_values is not None:
                    past_key_values = tuple((k.repeat(num_beams, 1, 1, 1), v.repeat(num_beams, 1, 1, 1)) for k, v in past_key_values)

            model_inputs = self.tokenizer([input_text], add_special_tokens=False, return_tensors="pt").to(self.device)

            curr_input_len = model_inputs.input_ids[0].size(0)

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

            if not is_read:
                past_key_values = prev_key_values

            response = response.replace(self.eos_token, self.eow_token)

            if is_read and self.eor_token in response:
                is_read = False
                input_text = f"{input_text}{response}"
            elif not is_read and (self.eow_token in response or generated_ids[0].size(0) >= self.gen_kwargs["max_new_tokens"]):
                is_read = True
                read_tok_num = 0
                hypo = response.rstrip().replace(self.eow_token, "")
                if self.eow_token not in response:
                    response = f"{response}{self.eow_token}"
                input_text = f"{input_text}{response}"
                preds.append(hypo)
            elif idx >= len(src_tokens):
                is_read = False
                input_text = f"{input_text}{self.eor_token}"
                past_key_values = prev_key_values
            elif is_read and len(generated_ids[0]) > 1:
                past_key_values = prev_key_values
            elif is_read and self.args.document_level and input_text[-1] in "。？！.!?" and read_tok_num >= 20:
                is_read = False
                input_text = f"{input_text}{self.eor_token}"

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

        while idx < len(src_tokens) or (not is_read):
            if is_read:
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

            response = response.replace(self.eos_token, self.eow_token)

            if is_read and self.eor_token in response:
                is_read = False
                input_text = f"{input_text}{response}"
            elif not is_read and (self.eow_token in response or generated_ids[0].size(0) >= self.gen_kwargs["max_new_tokens"]):
                hypo = response.rstrip().replace(self.eow_token, "")
                if self.eow_token not in response:
                    response = f"{response}{self.eow_token}"
                input_text = f"{input_text}{response}"
                preds.append(hypo)
                is_read = True
                read_tok_num = 0
            elif idx >= len(src_tokens):
                is_read = False
                input_text = f"{input_text}{self.eor_token}"
            elif is_read and self.args.document_level and input_text[-1] in "。？！.!?" and read_tok_num >= 20:
                is_read = False
                input_text = f"{input_text}{self.eor_token}"

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

            delays, _ = compute_delays(read_contents, hypo, src_lang, tgt_lang)
            AL = AverageLagging(delays, src_len, ref_len)
            LAAL = LengthAdaptiveAverageLagging(delays, src_len, ref_len)

            ALs.append(AL)
            LAALs.append(LAAL)

            prediction["delays"] = str(delays)
            prediction["BLEU"] = bleu
            prediction["AL"] = AL
            prediction["LAAL"] = LAAL

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prediction_file = os.path.join(output_dir, "prediction.json")
        with open(prediction_file, "w", encoding="utf8") as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=4)

        result_file = os.path.join(output_dir, "results.json")
        with open(result_file, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def simul_eval(self):
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
    args = load_infer_args()
    infer = SimulInference(args)
    infer.simul_eval()


if __name__ == "__main__":
    run_simuleval()
